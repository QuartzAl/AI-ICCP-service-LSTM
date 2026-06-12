#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - Final Model Production Training & Deployment Script
Reads optimized hyperparameters, trains the definitive LSTM model, and exports both 
the model architecture/weights (.keras) and the preprocessing scaler (.joblib).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib  # Standard for saving scalers
from datetime import datetime

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Machine Learning framework libraries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-LSTM-Production-Save")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

LOOKBACK_WINDOW = 45
PREDICTION_STEP = 15
BATCH_SIZE = 32
FINAL_TRAIN_EPOCHS = 50  # Giving it full runway to converge
HYPERPARAM_FILE = "best_hyperparameters.txt"

# --- Deployment Paths ---
ARTIFACT_DIR = "production_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(ARTIFACT_DIR, "iccp_lstm_model.keras")
SCALER_SAVE_PATH = os.path.join(ARTIFACT_DIR, "iccp_scaler.joblib")


def query_influxdb_data() -> pd.DataFrame:
    """Queries telemetry from InfluxDB."""
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -25d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 30s, fn: mean, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"])
    '''
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=3000)
        df = client.query_api().query_data_frame(query=flux_query)
        client.close()
        if df.empty: raise ValueError("Empty dataset.")
        df = df.rename(columns={"_time": "time"})
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df[["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]]
    except Exception as e:
        logger.error(f"InfluxDB data loading failed: {e}")
        return None

def create_sequences(data: np.ndarray, lookback: int, pred_step: int):
    X, y = [], []
    for i in range(len(data) - lookback - pred_step + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback + pred_step - 1, 2]) # Index 2 is electrode_V
    return np.array(X), np.array(y)

def parse_hyperparameters(filepath: str) -> dict:
    """Parses the exported best_hyperparameters.txt file into a usable Python dictionary."""
    if not os.path.exists(filepath):
        logger.error(f"Hyperparameter file '{filepath}' not found! Run the optimization script first.")
        sys.exit(1)
        
    hps = {}
    logger.info(f"Parsing optimized parameters from {filepath}...")
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line and "OPTIMIZED" not in line and "===" not in line:
                key, val = line.split(":")
                key = key.strip()
                val = val.strip()
                # Parse numeric types safely
                if key in ['lstm_units_1', 'lstm_units_2', 'dense_units_1']:
                    hps[key] = int(val)
                elif key in ['dropout_rate', 'learning_rate']:
                    hps[key] = float(val)
    return hps

def build_production_model(hps: dict) -> Sequential:
    """Builds the explicit network architecture using parsed hyperparameters."""
    num_features = 5
    model = Sequential([
        LSTM(units=hps['lstm_units_1'], return_sequences=True, input_shape=(LOOKBACK_WINDOW, num_features)),
        Dropout(hps['dropout_rate']),
        LSTM(units=hps['lstm_units_2'], return_sequences=False),
        Dropout(hps['dropout_rate']),
        Dense(units=hps['dense_units_1'], activation='relu'),
        Dense(units=1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=hps['learning_rate']),
        loss='mean_squared_error'
    )
    return model

def main():
    # 1. Load data and clean
    df = query_influxdb_data()
    if df is None: return
        
    df = df.ffill().bfill()
    df['future_target_current_mA'] = df['target_current_mA'].shift(-PREDICTION_STEP)
    df = df.dropna()
    
    features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "future_target_current_mA"]
    data_values = df[features].values 
    
    # Fit production Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW, PREDICTION_STEP)
    X, y = np.ascontiguousarray(X), np.ascontiguousarray(y)

    # 2. Parse optimized configs and build the model
    hps = parse_hyperparameters(HYPERPARAM_FILE)
    logger.info(f"Loaded config: {hps}")
    model = build_production_model(hps)

    # 3. Train on 100% of the available data for production deployment
    logger.info("Beginning production training run on full dataset...")
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.25, patience=2, min_lr=1e-7, verbose=1)
    
    model.fit(
        X, y,
        epochs=FINAL_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[lr_scheduler],
        verbose=1
    )
    
    # 4. Save Artifacts according to Best Practices
    logger.info("--- SAVING PRODUCTION ARTIFACTS ---")
    
    # Save the scaler
    try:
        joblib.dump(scaler, SCALER_SAVE_PATH)
        logger.info(f" [SUCCESS] Scaler exported to: '{SCALER_SAVE_PATH}'")
    except Exception as e:
        logger.error(f"Failed to save scaler: {e}")

    # Save the trained model
    try:
        # Saving in native Keras format contains architecture, weights, and compilation info
        model.save(MODEL_SAVE_PATH)
        logger.info(f" [SUCCESS] LSTM Model exported to: '{MODEL_SAVE_PATH}'")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    main()
