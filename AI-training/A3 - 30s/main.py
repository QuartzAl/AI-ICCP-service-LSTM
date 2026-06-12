#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - LSTM Predictive Modeling Script
Predicts future reference electrode potential (electrode_V) based on historical sequence data
combined with the context of future planned target currents.

Includes KerasTuner (Bayesian Optimization) for hyperparameter search and 
comprehensive evaluation metrics (RMSE, MAE, R2 Score).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Machine Learning framework libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-LSTM-FutureContext")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

LOOKBACK_WINDOW = 45
PREDICTION_STEP = 15
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
MAX_TUNER_EPOCHS = 10  # Keeping it short for tuning runs
FINAL_TRAIN_EPOCHS = 30
HYPERPARAM_FILE = "best_hyperparameters.txt"

def query_influxdb_data() -> pd.DataFrame:
    """Queries telemetry from InfluxDB and aggregates it."""
    logger.info(f"Querying InfluxDB at {INFLUXDB_URL} for device_id '{DEVICE_ID}'...")
    
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 30s, fn: mean, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"])
    '''
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=3000)
        query_api = client.query_api()
        df = query_api.query_data_frame(query=flux_query)
        client.close()
        
        if df.empty:
            raise ValueError("Query returned an empty dataset.")
            
        df = df.rename(columns={"_time": "time"})
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]
        return df[features]
    except Exception as e:
        logger.warning(f"Could not load live InfluxDB data: {e}")
        return None

def create_sequences(data: np.ndarray, lookback: int, pred_step: int):
    """Transforms raw arrays into 3D structural inputs for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback - pred_step + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback + pred_step - 1, 2]) # 2 is index of electrode_V
    return np.array(X), np.array(y)

def build_tunable_model(hp):
    """Model factory for KerasTuner to inject dynamic search ranges."""
    num_features = 5 
    
    model = Sequential()
    
    # Tunable Parameter 1: LSTM units for the first structural layer
    hp_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=32)
    model.add(LSTM(units=hp_units_1, return_sequences=True, input_shape=(LOOKBACK_WINDOW, num_features), recurrent_activation='hard_sigmoid'))
    
    # Tunable Parameter 2: Dropout rates
    hp_dropout = hp.Float('dropout_rate', min_value=0.05, max_value=0.35, step=0.05)
    model.add(Dropout(hp_dropout))
    
    # Tunable Parameter 3: LSTM units for the second sequence layer
    hp_units_2 = hp.Int('lstm_units_2', min_value=96, max_value=160, step=16)
    model.add(LSTM(units=hp_units_2, return_sequences=False, recurrent_activation='hard_sigmoid'))
    model.add(Dropout(hp_dropout))
    
    # Core reasoning layers
    hp_units_3 = hp.Int('dense_units_1', min_value=16, max_value=64, step=16)
    model.add(Dense(units=hp_units_3, activation='relu'))
    model.add(Dense(units=1))
    
    # Tunable Parameter 4: Learning rate paths
    hp_lr = hp.Choice('learning_rate', values=[5e-4, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=hp_lr),
        loss='mean_squared_error'
    )
    return model

def main():
    # 1. Pipeline execution and cleaning
    df = query_influxdb_data()
    if df is None:
        logger.error("Could not get influxdb data")
        return
        
    df = df.ffill().bfill()
    df['future_target_current_mA'] = df['target_current_mA'].shift(-PREDICTION_STEP)
    df = df.dropna()
    
    features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "future_target_current_mA"]
    data_values = df[features].values 
    num_features = data_values.shape[1]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, PREDICTION_STEP)
    logger.info(f"Generated input tensor dimensions: X: {X.shape}, y: {y.shape}")

    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = np.ascontiguousarray(X_train)
    y_train = np.ascontiguousarray(y_train)
    X_test = np.ascontiguousarray(X_test)
    y_test = np.ascontiguousarray(y_test)

    # 2. Configure the automated Hyperparameter Optimization Tuner
    logger.info("Initializing KerasTuner Bayesian Optimization Engine...")
    tuner = kt.BayesianOptimization(
        build_tunable_model,
        objective='val_loss',
        max_trials=20,                      # Total model configurations to test
        directory='kt_iccp_tuning_registry',
        project_name='lstm_setpoint_optimization',
        overwrite=True
    )

    # tuner = kt.Hyperband(
    #     build_tunable_model,
    #     objective='val_loss',
    #     directory='kt_iccp_tuning_hyperband',
    #     project_name='lstm_setpoint_optimization_hyperband',
    #     overwrite=True
    # )

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    # Run structural grid variations
    tuner.search(
        X_train, y_train,
        validation_split=0.1,
        epochs=MAX_TUNER_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 3. Retrieve, isolate, and save best performance models
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(f"--- OPTIMIZATION FOUND BEST PARAMETERS ---")
    logger.info(f"Best Layer 1 LSTM Units: {best_hps.get('lstm_units_1')}")
    logger.info(f"Best Layer 2 LSTM Units: {best_hps.get('lstm_units_2')}")
    logger.info(f"Best Network Dropout Rate: {best_hps.get('dropout_rate')}")
    logger.info(f"Best Network Dense Units: {best_hps.get('dense_units_1')}")
    logger.info(f"Best Optimized Learning Rate: {best_hps.get('learning_rate')}")
    
    # Save parameters cleanly to a text file
    try:
        with open(HYPERPARAM_FILE, "w") as f:
            f.write("==================================================\n")
            f.write(f"  OPTIMIZED LSTM HYPERPARAMETERS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("==================================================\n")
            f.write(f"lstm_units_1:  {best_hps.get('lstm_units_1')}\n")
            f.write(f"lstm_units_2:  {best_hps.get('lstm_units_2')}\n")
            f.write(f"dropout_rate:  {best_hps.get('dropout_rate'):.4f}\n")
            f.write(f"dense_units_1: {best_hps.get('dense_units_1')}\n")
            f.write(f"learning_rate: {best_hps.get('learning_rate')}\n")
            f.write("==================================================\n")
            f.write(f"Root Mean Squared Error (RMSE) : {rmse:.4f} V  (Penalizes large blunders)")
            f.write(f"Mean Absolute Error (MAE)       : {mae:.4f} V  (Average straight deviation)")
            f.write(f"Coefficient of Determination(R²): {r2:.4f}     (Goal: Close to 1.0)")
            f.write(f"Lookback window:                : {LOOKBACK_WINDOW}")
            f.write(f"Prediction step:                : {PREDICTION_STEP}")
        logger.info(f"Successfully saved optimized parameters to '{HYPERPARAM_FILE}'")
    except Exception as e:
        logger.error(f"Failed to write hyperparameters text file: {e}")
        
    # 4. Train the absolute best model configuration fully
    logger.info("Rebuilding and training final model using top parameters...")
    model = tuner.hypermodel.build(best_hps)
    
    final_early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=2, min_lr=1e-7, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=FINAL_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[final_early_stopping, lr_scheduler],
        verbose=1
    )
    
    # 5. Comprehensive Model Evaluation & Metric Logging
    logger.info("Evaluating final model on test data...")
    predictions_scaled = model.predict(X_test)
    
    # Invert predictions back to scale
    temp_pred_matrix = np.zeros((len(predictions_scaled), num_features))
    temp_pred_matrix[:, 2] = predictions_scaled.flatten() 
    predictions_real = scaler.inverse_transform(temp_pred_matrix)[:, 2]
    
    # Invert target actuals back to scale
    temp_actual_matrix = np.zeros((len(y_test), num_features))
    temp_actual_matrix[:, 2] = y_test
    actual_real = scaler.inverse_transform(temp_actual_matrix)[:, 2]
    
    # Calculate performance scores
    rmse = np.sqrt(np.mean((predictions_real - actual_real) ** 2))
    mae = mean_absolute_error(actual_real, predictions_real)
    r2 = r2_score(actual_real, predictions_real)
    
    # Log everything out cleanly to evaluate if the model is good
    logger.info("==========================================================")
    logger.info("                   FINAL PERFORMANCE METRICS               ")
    logger.info("==========================================================")
    logger.info(f"Root Mean Squared Error (RMSE) : {rmse:.4f} V  (Penalizes large blunders)")
    logger.info(f"Mean Absolute Error (MAE)       : {mae:.4f} V  (Average straight deviation)")
    logger.info(f"Coefficient of Determination(R²): {r2:.4f}     (Goal: Close to 1.0)")
    logger.info("==========================================================")

    # 6. Render outputs
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    
    axes[0].plot(history.history['loss'], label='Training Loss (MSE)', color='#3b82f6', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#f59e0b', linewidth=2)
    axes[0].set_title("Optimized LSTM Training Process Loss Profile", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    test_dates = df.index[-len(y_test):]
    axes[1].plot(test_dates, actual_real, label='Actual Reference Potential', color='#10b981', linewidth=2)
    axes[1].plot(test_dates, predictions_real, label='LSTM Tuned Forecast', color='#ef4444', linestyle='--', linewidth=2)
    axes[1].set_title(f"Actual vs Prediction Comparison (MAE: {mae:.4f})", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Electrode_V")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_img = "iccp_lstm_optimized_results.png"
    plt.savefig(output_img)
    logger.info(f"Analysis saved as: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()