#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - LSTM Predictive Modeling Script
Predicts future reference electrode potential (electrode_V) based on historical sequence data
combined with the context of future planned target currents.

Dependencies to install:
    pip install influxdb-client pandas numpy scikit-learn tensorflow matplotlib
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-LSTM-FutureContext")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

# --- LSTM & Horizon Configurations ---
# At 10-minute intervals:
# 144 steps = 24 hours of history (144 * 10 mins)
# Prediction step of 6 = predict 1 hour into the future (6 * 10 mins)
LOOKBACK_WINDOW = 96      # Number of historical 10-minute steps to look back (24 hours)
PREDICTION_STEP = 96        # Number of 10-minute steps in the future to predict (1 hour)
TRAIN_SPLIT = 0.8         # 80% train, 20% test
BATCH_SIZE = 32
EPOCHS = 20

def query_influxdb_data() -> pd.DataFrame:
    """
    Connects to InfluxDB, executes a Flux query to retrieve telemetry for device_id '001',
    aggregates data into 10-minute mean windows, and parses them into a Pandas DataFrame.
    """
    logger.info(f"Querying InfluxDB at {INFLUXDB_URL} for device_id '{DEVICE_ID}' with 10-minute aggregation...")
    
    # We query the last 30 days of telemetry data.
    # We include 'target_current_mA' as the setpoint feature.
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 2m, fn: mean, createEmpty: true)
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
            
        # Clean up database columns
        df = df.rename(columns={"_time": "time"})
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Ensure we keep the target current feature
        features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]
        df = df[features]
        
        logger.info(f"Successfully loaded {len(df)} 10-minute aggregated records from InfluxDB.")
        return df
        
    except Exception as e:
        logger.warning(f"Could not load live InfluxDB data: {e}")
        return None



def create_sequences(data: np.ndarray, lookback: int, pred_step: int):
    """
    Transforms clean 2D array data into 3D input sequences for the LSTM layer.
    
    Returns:
        X: Sequence windows with shape [samples, time_steps, features]
        y: Target label vector containing future 'electrode_V' values
    """
    X, y = [], []
    for i in range(len(data) - lookback - pred_step + 1):
        # Extract sliding historical window of all features (including future context at each step)
        X.append(data[i:(i + lookback)])
        # Target index 2 represents 'electrode_V' (reference potential) at the prediction step
        y.append(data[i + lookback + pred_step - 1, 2])
        
    return np.array(X), np.array(y)


def main():
    # 1. Gather Telemetry Dataset (Now using 10m aggregation)
    df = query_influxdb_data()
    if df is None:
        logger.error("Could not get influxdb data")
        return
    
    # Fill any sparse gaps from database query drops safely
    df = df.ffill().bfill()
    
    # =========================================================================
    # FUTURE SETPOINT CONTEXT ALIGNMENT (THE CORE CHANGE)
    # Shift target_current_mA backwards by the prediction horizon.
    # At index t, 'future_target_current_mA' is exactly target_current_mA at t + PREDICTION_STEP.
    # Since we know the planned current in advance, this is safe to use as an input.
    # =========================================================================
    df['future_target_current_mA'] = df['target_current_mA'].shift(-PREDICTION_STEP)
    
    # Drop the final rows where future context is missing due to the shift
    df = df.dropna()
    
    # 2. Extract Data Values
    # Features order:
    # 0: bus_voltage_V
    # 1: current_mA (measured/historical)
    # 2: electrode_V (measured/historical target)
    # 3: soil_humidity_V (environmental/historical)
    # 4: future_target_current_mA (planned future setpoint context)
    features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "future_target_current_mA"]
    data_values = df[features].values 
    num_features = data_values.shape[1]
    
    # 3. Scale inputs between [0, 1] for stable backpropagation in the LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)
    
    # 4. Generate sequences
    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, PREDICTION_STEP)
    logger.info(f"Generated input tensor dimensions: X: {X.shape}, y: {y.shape}")
    
    # 5. Partition into Training and Test splits
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training split: {X_train.shape[0]} samples. Testing split: {X_test.shape[0]} samples.")
    
    # 6. Initialize LSTM Model Architecture
    logger.info("Configuring LSTM Network Structure with Future Setpoint Context...")
    model = Sequential([
        # First LSTM Layer
        LSTM(units=64, return_sequences=True, input_shape=(LOOKBACK_WINDOW, num_features)),
        Dropout(0.1),
        
        # Second LSTM Layer
        LSTM(units=32, return_sequences=False),
        Dropout(0.1),
        
        # Fully connected regression layers
        Dense(units=16, activation='relu'),
        Dense(units=1) # Singular output scalar: Predicted Electrode Potential (V)
    ])
    
    # Compile model using Mean Squared Error loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    # Prevent overtraining using EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
    # 7. Model Training
    logger.info("Beginning model training cycles...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 8. Model Evaluation & Inference
    logger.info("Evaluating predictions against testing matrix...")
    predictions_scaled = model.predict(X_test)
    
    # Denormalize predictions back to natural physical values
    temp_pred_matrix = np.zeros((len(predictions_scaled), num_features))
    temp_pred_matrix[:, 2] = predictions_scaled.flatten() # 2 is the index of electrode_V
    predictions_real = scaler.inverse_transform(temp_pred_matrix)[:, 2]
    
    # Denormalize actual results
    temp_actual_matrix = np.zeros((len(y_test), num_features))
    temp_actual_matrix[:, 2] = y_test
    actual_real = scaler.inverse_transform(temp_actual_matrix)[:, 2]
    
    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((predictions_real - actual_real) ** 2))
    logger.info(f"Model Validation Complete. Root Mean Squared Error (RMSE): {rmse:.4f} V")
    
    # =========================================================================
    # WHAT-IF SCENARIO SIMULATION
    # Prove the future setpoint context works by running a dynamic simulation.
    # We will copy a test sequence and force its future target currents to be different.
    # =========================================================================
    logger.info("Running What-If Scenario Simulations...")
    sample_seq = X_test[-1].copy() # Grab the very last test sequence
    
    # Scenario A: Let's set the future target current to a low value (50 mA)
    seq_low_current = sample_seq.copy()
    # 4 is the index of future_target_current_mA
    seq_low_current[:, 4] = scaler.transform(np.array([[0, 0, 0, 0, 50.0]]))[0, 4] 
    pred_low_scaled = model.predict(np.expand_dims(seq_low_current, axis=0))
    
    temp_low = np.zeros((1, num_features))
    temp_low[0, 2] = pred_low_scaled[0, 0]
    pred_low_V = scaler.inverse_transform(temp_low)[0, 2]

    # Scenario B: Let's set the future target current to a high value (250 mA)
    seq_high_current = sample_seq.copy()
    seq_high_current[:, 4] = scaler.transform(np.array([[0, 0, 0, 0, 250.0]]))[0, 4]
    pred_high_scaled = model.predict(np.expand_dims(seq_high_current, axis=0))
    
    temp_high = np.zeros((1, num_features))
    temp_high[0, 2] = pred_high_scaled[0, 0]
    pred_high_V = scaler.inverse_transform(temp_high)[0, 2]
    
    logger.info("--- WHAT-IF FORECAST RESULTS ---")
    logger.info(f"Scenario 1 (Low target current of 50 mA)  -> Predicted Electrode Potential: {pred_low_V:.3f} V")
    logger.info(f"Scenario 2 (High target current of 250 mA) -> Predicted Electrode Potential: {pred_high_V:.3f} V")
    logger.info("--------------------------------")

    # 9. Plot the output figures
    logger.info("Rendering output prediction figures...")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    
    # Plot A: Model Training Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss (MSE)', color='#3b82f6', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#f59e0b', linewidth=2)
    axes[0].set_title("LSTM CP Predictor with Future Setpoint Context - Training Metrics", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot B: Actual vs Predicted values
    test_dates = df.index[-len(y_test):]
    axes[1].plot(test_dates, actual_real, label=f'Actual Reference Potential', color='#10b981', linewidth=2)
    axes[1].plot(test_dates, predictions_real, label=f'LSTM Forecast with Setpoint Context', color='#ef4444', linestyle='--', linewidth=2)
    axes[1].set_title(f"Electrode Potential Prediction (RMSE: {rmse:.4f} V)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Structure-to-Soil Potential (Electrode_V)")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save chart output and display
    output_img = "iccp_lstm_results.png"
    plt.savefig(output_img)
    logger.info(f"Analysis saved as: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()
