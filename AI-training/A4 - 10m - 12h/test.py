#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - LSTM Predictive Model Evaluator
Trains the LSTM model under fixed/optimized configurations (including soil humidity)
and calculates industrial validation metrics: RMSE, MAE, cvRMSE, MDA, MAPE, and R2.
Saves comprehensive metrics and training parameter summaries to an evaluation report,
exports all actual vs. predicted values sequentially to a CSV file, and saves both 
the trained model (.keras) and scaler normalizer (.joblib) in a designated directory.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib  # Scaler serialization
import matplotlib.pyplot as plt
from datetime import datetime

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Machine Learning framework libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-LSTM-Evaluator")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

LOOKBACK_WINDOW = 90
PREDICTION_STEP = 72
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
EVAL_TRAIN_EPOCHS = 40
HYPERPARAM_FILE = "best_hyperparameters.txt"

# --- Folder and Artifact Storage Configuration ---
OUTPUT_DIR = "evaluation_artifacts"

# Relocate all generated files into the output directory
REPORT_FILE = os.path.join(OUTPUT_DIR, "model_evaluation_report.txt")
PLOT_FILE = os.path.join(OUTPUT_DIR, "evaluation_comparison.png")
CSV_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_predictions.csv")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "iccp_lstm_model.keras")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "iccp_scaler.joblib")

# Target 5 features (soil humidity included)
FEATURES = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "future_target_current_mA"]
NUM_FEATURES = len(FEATURES)

# Standard Fixed Default Hyperparameters (Fallback if best_hyperparameters.txt doesn't exist)
DEFAULT_HPS = {
    "lstm_units_1": 128,
    "lstm_units_2": 128,
    "dropout_rate": 0.15,
    "dense_units_1": 32,
    "learning_rate": 0.0005
}

def query_influxdb_data() -> pd.DataFrame:
    """Queries telemetry from InfluxDB including soil humidity."""
    logger.info(f"Querying InfluxDB at {INFLUXDB_URL} for device_id '{DEVICE_ID}'...")
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -25d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 10m, fn: mean, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"])
    '''
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=3000)
        df = client.query_api().query_data_frame(query=flux_query)
        client.close()
        
        if df.empty:
            raise ValueError("Query returned an empty dataset.")
            
        df = df.rename(columns={"_time": "time"})
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df[["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]]
    except Exception as e:
        logger.error(f"InfluxDB data loading failed: {e}")
        return None

def create_sequences(data: np.ndarray, lookback: int, pred_step: int):
    """Transforms raw arrays into 3D structural inputs for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback - pred_step + 1):
        X.append(data[i:(i + lookback)])
        # Index 2 of FEATURES corresponds to electrode_V
        y.append(data[i + lookback + pred_step - 1, 2])
    return np.array(X), np.array(y)

def get_hyperparameters() -> dict:
    """Loads hyperparams from 'best_hyperparameters.txt' if available, otherwise falls back to defaults."""
    hps = DEFAULT_HPS.copy()
    if os.path.exists(HYPERPARAM_FILE):
        logger.info(f"Found '{HYPERPARAM_FILE}'. Loading custom optimized hyperparameters...")
        try:
            with open(HYPERPARAM_FILE, "r") as f:
                for line in f:
                    if ":" in line and "OPTIMIZED" not in line and "===" not in line:
                        key, val = line.split(":")
                        key = key.strip()
                        val = val.strip()
                        if key in ['lstm_units_1', 'lstm_units_2', 'dense_units_1']:
                            hps[key] = int(val)
                        elif key in ['dropout_rate', 'learning_rate']:
                            hps[key] = float(val)
            logger.info("Successfully loaded parameters from file.")
        except Exception as e:
            logger.warning(f"Error parsing {HYPERPARAM_FILE}, using hardcoded defaults. Error: {e}")
    else:
        logger.info(f"'{HYPERPARAM_FILE}' not found. Using fixed default hyperparameters: {hps}")
    return hps

def build_model(hps: dict) -> Sequential:
    """Builds the explicit LSTM architecture using specified hyperparameters."""
    model = Sequential([
        LSTM(units=hps['lstm_units_1'], return_sequences=True, input_shape=(LOOKBACK_WINDOW, NUM_FEATURES), recurrent_activation="hard_sigmoid"),
        Dropout(hps['dropout_rate']),
        LSTM(units=hps['lstm_units_2'], return_sequences=False, recurrent_activation="hard_sigmoid"),
        Dropout(hps['dropout_rate']),
        Dense(units=hps['dense_units_1'], activation='relu'),
        Dense(units=1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=hps['learning_rate']),
        loss='mean_squared_error'
    )
    return model

def calculate_mda(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates Mean Directional Accuracy (MDA).
    Compares the forecasted direction of change (up/down) to the actual change direction.
    """
    actual_diff = np.sign(actual[1:] - actual[:-1])
    predicted_diff = np.sign(predicted[1:] - actual[:-1])
    valid_indices = (actual_diff != 0)
    if not np.any(valid_indices):
        return 0.0
    return np.mean(actual_diff[valid_indices] == predicted_diff[valid_indices])

def main():
    # Ensure the output artifacts directory exists before saving anything
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Pipeline Execution and Preprocessing
    df = query_influxdb_data()
    if df is None:
        return
        
    # Drop rows that are entirely NaN before performing any fill operations
    # This prevents forward-filling over massive blocks of empty connection/downtime gaps
    df = df.dropna(how='all')
    
    # Fill remaining small gaps
    df = df.ffill().bfill()
    df['future_target_current_mA'] = df['target_current_mA'].shift(-PREDICTION_STEP)
    df = df.dropna()
    
    data_values = df[FEATURES].values 
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, PREDICTION_STEP)
    
    # Establish complete sequence timeline timestamps
    sequence_dates = df.index[LOOKBACK_WINDOW + PREDICTION_STEP - 1:]
    
    # Train-test split (sequential partition for time series)
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)
    X_test = np.ascontiguousarray(X_test)
    y_test = np.ascontiguousarray(y_test)
    
    # Split the datetimes to align precisely with training and test subsets
    train_dates = sequence_dates[:split_idx]
    test_dates = sequence_dates[split_idx:]

    # 2. Setup Hyperparameters and Build Model
    hps = get_hyperparameters()
    model = build_model(hps)
    
    # 3. Model Training with EarlyStopping and validation monitoring
    logger.info("Training the evaluation model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=2, min_lr=1e-7, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EVAL_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # 4. Predictions & Rescaling
    logger.info("Evaluating and rescaling complete timeline predictions...")
    predictions_train_scaled = model.predict(X_train)
    predictions_test_scaled = model.predict(X_test)
    
    # Inverse transform train predictions
    temp_train = np.zeros((len(predictions_train_scaled), NUM_FEATURES))
    temp_train[:, 2] = predictions_train_scaled.flatten()
    train_predictions_real = scaler.inverse_transform(temp_train)[:, 2]
    
    # Inverse transform test predictions
    temp_test = np.zeros((len(predictions_test_scaled), NUM_FEATURES))
    temp_test[:, 2] = predictions_test_scaled.flatten() 
    predictions_real = scaler.inverse_transform(temp_test)[:, 2]
    
    # Inverse transform entire target actual set (y)
    temp_actual_all = np.zeros((len(y), NUM_FEATURES))
    temp_actual_all[:, 2] = y
    actual_all_real = scaler.inverse_transform(temp_actual_all)[:, 2]
    
    # Isolate test ground truth actuals for metrics calculation
    actual_real = actual_all_real[split_idx:]

    # 5. Comprehensive Metric Computations (Out-of-sample Test Data)
    rmse = np.sqrt(mean_squared_error(actual_real, predictions_real))
    mae = mean_absolute_error(actual_real, predictions_real)
    
    # cvRMSE: Coefficient of Variation of the RMSE
    actual_mean = np.mean(actual_real)
    cv_rmse = (rmse / np.abs(actual_mean)) * 100 if actual_mean != 0 else 0.0
    
    # MDA: Mean Directional Accuracy
    mda = calculate_mda(actual_real, predictions_real) * 100
    
    # R2: Coefficient of Determination
    r2 = r2_score(actual_real, predictions_real)
    
    # MAPE: Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual_real - predictions_real) / actual_real)) * 100 if not np.any(actual_real == 0) else 0.0
    
    # Max Absolute Error
    max_err = np.max(np.abs(actual_real - predictions_real))

    # Log metrics directly to the logger console
    logger.info("==========================================================")
    logger.info("               VALIDATION PERFORMANCE REPORT              ")
    logger.info("==========================================================")
    logger.info(f"Root Mean Squared Error (RMSE)    : {rmse:.5f} V")
    logger.info(f"Mean Absolute Error (MAE)          : {mae:.5f} V")
    logger.info(f"Coefficient of Variation (cvRMSE)  : {cv_rmse:.3f}%")
    logger.info(f"Mean Directional Accuracy (MDA)    : {mda:.2f}%")
    logger.info(f"Mean Absolute Percentage Error(MAPE): {mape:.3f}%")
    logger.info(f"Max Absolute Deviation Error       : {max_err:.5f} V")
    logger.info(f"Coefficient of Determination (R²)  : {r2:.4f}")
    logger.info("==========================================================")

    # 6. Save Model and Normalizer Scaler to Disk (New Step)
    logger.info("Saving trained model and preprocessing scaler to destination directory...")
    try:
        # Save model inside native .keras zip package
        model.save(MODEL_SAVE_PATH)
        logger.info(f" [SUCCESS] Keras model saved successfully to: '{MODEL_SAVE_PATH}'")
        
        # Save Scaler using joblib
        joblib.dump(scaler, SCALER_SAVE_PATH)
        logger.info(f" [SUCCESS] Preprocessing scaler saved successfully to: '{SCALER_SAVE_PATH}'")
    except Exception as e:
        logger.error(f"Failed to export neural model/scaler artifacts: {e}")

    # 7. Save actual and predicted values to CSV
    logger.info("Saving predictions and ground truth values to CSV...")
    try:
        # Build separate DataFrames for train and test partitions to maintain correct timeline alignment
        df_train_preds = pd.DataFrame({
            'actual_electrode_V': actual_all_real[:split_idx],
            'predicted_electrode_V': train_predictions_real,
            'partition': 'train'
        }, index=train_dates)

        df_test_preds = pd.DataFrame({
            'actual_electrode_V': actual_real,
            'predicted_electrode_V': predictions_real,
            'partition': 'test'
        }, index=test_dates)

        # Concatenate train and test chronologically
        df_combined_preds = pd.concat([df_train_preds, df_test_preds])
        df_combined_preds.index.name = 'time'
        
        df_combined_preds.to_csv(CSV_OUTPUT_FILE)
        logger.info(f"Predictions and actual data successfully saved to '{CSV_OUTPUT_FILE}'")
    except Exception as e:
        logger.error(f"Failed to export predictions to CSV: {e}")

    # 8. Save Hyperparameters & Performance metrics to Report File
    try:
        with open(REPORT_FILE, "w") as f:
            f.write("========================================================================\n")
            f.write(f"           ICCP TELEMETRY LSTM MODEL EVALUATION REPORT\n")
            f.write(f"           Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("========================================================================\n\n")
            
            f.write("--- TELEMETRY DATA STRUCTURE CONFIGURATION ---\n")
            f.write(f"Device ID Selected           : {DEVICE_ID}\n")
            f.write(f"Input Lookback Sequence Steps: {LOOKBACK_WINDOW} steps (90 mins context)\n")
            f.write(f"Target Future Prediction Step: {PREDICTION_STEP} steps (30 mins forward)\n")
            f.write(f"Features Modeled             : {', '.join(FEATURES)}\n")
            f.write(f"Train/Test Partition Split   : {TRAIN_SPLIT*100}% Train / {(1-TRAIN_SPLIT)*100:.0f}% Test\n\n")

            f.write("--- TRAINED MODEL HYPERPARAMETERS ---\n")
            f.write(f"LSTM Structural Layer 1 Units : {hps['lstm_units_1']}\n")
            f.write(f"LSTM Structural Layer 2 Units : {hps['lstm_units_2']}\n")
            f.write(f"Dense Connectivity Layer Units: {hps['dense_units_1']}\n")
            f.write(f"Dropout Regularization Rate   : {hps['dropout_rate']:.4f}\n")
            f.write(f"Optimized Optimizer Base LR   : {hps['learning_rate']:.6f}\n")
            f.write(f"Batch Execution Training Size : {BATCH_SIZE}\n\n")

            f.write("--- DETAILED VALIDATION METRICS (OUT-OF-SAMPLE) ---\n")
            f.write(f"Root Mean Squared Error (RMSE)      : {rmse:.6f} V\n")
            f.write(f"Mean Absolute Error (MAE)            : {mae:.6f} V\n")
            f.write(f"Coefficient of Variation of RMSE (cvRMSE): {cv_rmse:.4f}%\n")
            f.write(f"Mean Directional Accuracy (MDA)      : {mda:.4f}%\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%\n")
            f.write(f"Max Scale Outlier Error (Max Error)  : {max_err:.6f} V\n")
            f.write(f"Coefficient of Determination (R²)    : {r2:.6f}\n\n")
            
            f.write("--- FILE EXPORT ARTIFACT PATHS ---\n")
            f.write(f"Keras Saved Model Directory          : {MODEL_SAVE_PATH}\n")
            f.write(f"MinMaxScaler Joblib Path             : {SCALER_SAVE_PATH}\n")
            f.write(f"Raw Predictions Output CSV           : {CSV_OUTPUT_FILE}\n")
            f.write("========================================================================\n")
        logger.info(f"Model validation details successfully saved to '{REPORT_FILE}'")
    except Exception as e:
        logger.error(f"Failed to generate evaluation report: {e}")

    # 9. Generate and Save Visual Output Comparison Plots
    try:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        
        # Loss Convergence profile
        axes[0].plot(history.history['loss'], label='Training Loss (MSE)', color='#3b82f6', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#f59e0b', linewidth=2)
        axes[0].set_title("LSTM Convergence Profile (Loss History)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Actual vs Forecast Comparison (Plotting the entire timeline)
        # Plot 1: The true target electrode_V actual values across the entire dataset timeline
        axes[1].plot(sequence_dates, actual_all_real, label='Actual Reference Potential (electrode_V)', color='#10b981', linewidth=2, alpha=0.9)
        
        # Plot 2: Overlay the in-sample predictions on the training data 
        axes[1].plot(train_dates, train_predictions_real, label='LSTM Training Fit (In-Sample)', color='#3b82f6', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Plot 3: Overlay the out-of-sample predictions on the testing data
        axes[1].plot(test_dates, predictions_real, label='LSTM Test Forecast (Out-of-Sample)', color='#ef4444', linestyle='--', linewidth=2.5)
        
        # Plot 4: Place a visual barrier at the split between the training and testing phases
        divider_date = test_dates[0]
        axes[1].axvline(x=divider_date, color='#4b5563', linestyle='-.', alpha=0.7, label='Train-Test Split Barrier')
        
        # Updated Title: Displaying only MAE and cvRMSE as requested
        axes[1].set_title(f"Telemetry Forecast vs. Real Actuals (Entire Dataset)\n(MAE: {mae:.5f} V, cvRMSE: {cv_rmse:.2f}%)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Electrode Potential (V)")
        axes[1].set_xlabel("Datetime Timeline")
        axes[1].legend(loc='best')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(PLOT_FILE)
        logger.info(f"Comparison graph exported successfully as: '{PLOT_FILE}'")
        plt.show()
    except Exception as e:
        logger.error(f"Failed to compile metric visuals: {e}")

if __name__ == "__main__":
    main()