#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - TFT Evaluation Script
Reads hyperparameters from a text file, trains the model, and evaluates it 
using both Rolling Forecast and Autoregressive Forecast methodologies.
Saves the final trained model to disk.
"""

import os
import sys
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Metrics and Scaling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from pytorch_lightning.callbacks import EarlyStopping

# Darts
from darts import TimeSeries, concatenate
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler

# Optimize precision
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-TFT-Eval")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

TRAIN_SPLIT = 0.80
HYPERPARAMS_FILE = "timeseries_models_metrics.txt"
MODEL_SAVE_PATH = "tft_optimal_model.pt"

def load_hyperparameters(filepath):
    """Parses the generated text file to extract the best TFT parameters."""
    if not os.path.exists(filepath):
        logger.error(f"Hyperparameters file '{filepath}' not found! Run the optimization script first.")
        sys.exit(1)
        
    logger.info(f"Parsing hyperparameters from '{filepath}'...")
    with open(filepath, "r") as f:
        content = f.read()
        
    params = {}
    int_keys = ['input_chunk_length', 'output_chunk_length', 'hidden_size', 'lstm_layers', 'num_attention_heads', 'batch_size', 'epochs_limit']
    float_keys = ['dropout', 'lr']
    
    for key in int_keys:
        match = re.search(fr"-\s*{key}:\s*(\d+)", content)
        if match: params[key] = int(match.group(1))
            
    for key in float_keys:
        match = re.search(fr"-\s*{key}:\s*([0-9.]+)", content)
        if match: params[key] = float(match.group(1))
            
    logger.info(f"Successfully loaded params: {params}")
    return params

def query_influxdb_data() -> pd.DataFrame:
    logger.info(f"Querying InfluxDB at {INFLUXDB_URL} for device_id '{DEVICE_ID}'...")
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -25d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 2m, fn: mean, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"])
    '''
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=3000)
    df = client.query_api().query_data_frame(query=flux_query)
    client.close()
    
    df = df.rename(columns={"_time": "time"})
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df.set_index('time', inplace=True)
    return df[["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]]

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mean_actual = np.mean(actual)
    cv_rmse = (rmse / mean_actual) * 100 if mean_actual != 0 else 0.0
    nmbe = (np.sum(actual - predicted) / (len(actual) * mean_actual)) * 100 if mean_actual != 0 else 0.0
    r2 = r2_score(actual, predicted)
    
    actual_diff = np.sign(actual[1:] - actual[:-1])
    pred_diff = np.sign(predicted[1:] - actual[:-1])
    mda = np.mean(actual_diff == pred_diff) * 100
    
    return {"MAE": mae, "cvRMSE": cv_rmse, "NMBE": nmbe, "MDA": mda, "R2": r2}

def reconstruct_rolling_absolute(pred_deltas, df_abs, time_index, chunk_out):
    """Reconstructs absolute values chunk-by-chunk using actuals as the rolling base."""
    pred_abs = []
    for i in range(0, len(pred_deltas), chunk_out):
        chunk_deltas = pred_deltas[i : i+chunk_out]
        current_time = time_index[i]
        abs_idx = df_abs.index.get_loc(current_time)
        base_value = df_abs['electrode_V'].iloc[abs_idx - 1]
        chunk_abs = base_value + np.cumsum(chunk_deltas)
        pred_abs.extend(chunk_abs)
    return np.array(pred_abs)

def reconstruct_autoregressive_absolute(pred_deltas, last_train_abs_value):
    """Reconstructs absolute values by cumulating ALL deltas off a single baseline (the last train value)."""
    return last_train_abs_value + np.cumsum(pred_deltas)

def rolling_forecast(model, full_target_scaled, train_idx, test_length, past_cov, future_cov, chunk_out):
    preds_list = []
    for i in range(0, test_length, chunk_out):
        context_end_idx = train_idx + i
        current_context = full_target_scaled[:context_end_idx]
        steps_to_predict = min(chunk_out, test_length - i)
        
        pred = model.predict(n=steps_to_predict, series=current_context, past_covariates=past_cov, future_covariates=future_cov, verbose=False)
        preds_list.append(pred)
    return concatenate(preds_list)

def main():
    # 1. Load Parameters
    hps = load_hyperparameters(HYPERPARAMS_FILE)
    
    # 2. Data Fetch & Differencing
    df_abs = query_influxdb_data()
    df_abs = df_abs.asfreq('2min').ffill().bfill().dropna()
    
    logger.info("Applying first-order differencing...")
    df_diff = df_abs.diff().dropna()
    
    target_col = "electrode_V"
    past_cov_cols = ["bus_voltage_V", "current_mA", "soil_humidity_V"]
    future_cov_cols = ["target_current_mA"] 
    
    target_ts = TimeSeries.from_dataframe(df_diff, value_cols=target_col)
    past_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=past_cov_cols)
    future_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=future_cov_cols)
    
    train_idx = int(len(df_diff) * TRAIN_SPLIT)
    
    # Target
    train_target = target_ts[:train_idx]
    test_target = target_ts[train_idx:]
    
    # Covariates
    train_past_cov = past_cov_ts[:train_idx]
    test_past_cov = past_cov_ts[train_idx:]
    train_future_cov = future_cov_ts[:train_idx]
    test_future_cov = future_cov_ts[train_idx:]
    
    # 3. Scaling
    logger.info("Scaling features...")
    target_scaler = Scaler(StandardScaler())
    past_cov_scaler = Scaler(StandardScaler())
    future_cov_scaler = Scaler(StandardScaler())
    
    target_scaler.fit(train_target)
    past_cov_scaler.fit(train_past_cov)
    future_cov_scaler.fit(train_future_cov)
    
    full_target_scaled = target_scaler.transform(target_ts)
    train_target_scaled = full_target_scaled[:train_idx]
    test_target_scaled = full_target_scaled[train_idx:]
    
    full_past_cov_scaled = past_cov_scaler.transform(past_cov_ts)
    train_past_cov_scaled = full_past_cov_scaled[:train_idx]
    test_past_cov_scaled = full_past_cov_scaled[train_idx:]
    
    full_future_cov_scaled = future_cov_scaler.transform(future_cov_ts)
    train_future_cov_scaled = full_future_cov_scaled[:train_idx]
    test_future_cov_scaled = full_future_cov_scaled[train_idx:]
    
    # Ground Truths
    y_test_delta = test_target.values().flatten()
    test_length = len(y_test_delta)
    y_test_abs = df_abs.loc[test_target.time_index, target_col].values
    last_train_abs = df_abs.iloc[train_idx][target_col] # The base for autoregressive
    
    # 4. Build and Train Model
    logger.info("Rebuilding TFT Model from loaded hyperparameters...")
    early_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")
    
    tft_model = TFTModel(
        input_chunk_length=hps.get('input_chunk_length', 60),
        output_chunk_length=hps.get('output_chunk_length', 15),
        hidden_size=hps.get('hidden_size', 32),
        lstm_layers=hps.get('lstm_layers', 1),
        num_attention_heads=hps.get('num_attention_heads', 4),
        dropout=hps.get('dropout', 0.1),
        batch_size=hps.get('batch_size', 32),
        n_epochs=hps.get('epochs_limit', 150),
        add_relative_index=True, 
        add_encoders={'cyclic': {'future': ['hour', 'minute']}}, 
        random_state=42,
        optimizer_kwargs={'lr': hps.get('lr', 1e-3)},
        likelihood=None,                      
        loss_fn=nn.HuberLoss(),               
        pl_trainer_kwargs={"devices": 1, "callbacks": [early_stopper]}      
    )
    
    logger.info("Training Model...")
    tft_model.fit(
        series=train_target_scaled,
        past_covariates=train_past_cov_scaled,
        future_covariates=train_future_cov_scaled,
        val_series=test_target_scaled,                     
        val_past_covariates=test_past_cov_scaled,          
        val_future_covariates=test_future_cov_scaled,      
        verbose=True
    )
    
    # 5. Evaluate: Rolling Forecast
    logger.info("Executing Rolling Forecast (simulating live production updates)...")
    rolling_scaled = rolling_forecast(tft_model, full_target_scaled, train_idx, test_length, full_past_cov_scaled, full_future_cov_scaled, hps.get('output_chunk_length', 15))
    rolling_delta = target_scaler.inverse_transform(rolling_scaled).values().flatten()
    rolling_abs = reconstruct_rolling_absolute(rolling_delta, df_abs, test_target.time_index, hps.get('output_chunk_length', 15))
    
    # 6. Evaluate: Autoregressive Forecast
    logger.info("Executing Autoregressive Forecast (blind prediction over entire horizon)...")
    ar_scaled = tft_model.predict(n=test_length, series=train_target_scaled, past_covariates=full_past_cov_scaled, future_covariates=full_future_cov_scaled, verbose=False)
    ar_delta = target_scaler.inverse_transform(ar_scaled).values().flatten()
    ar_abs = reconstruct_autoregressive_absolute(ar_delta, last_train_abs)
    
    # 7. Print Statistics
    metrics_rolling = calculate_metrics(y_test_abs, rolling_abs)
    metrics_ar = calculate_metrics(y_test_abs, ar_abs)
    
    logger.info("\n=========================================================")
    logger.info("                FINAL EVALUATION METRICS                 ")
    logger.info("=========================================================")
    logger.info("--- Rolling Forecast (Updating Base) ---")
    logger.info(f"MAE    : {metrics_rolling['MAE']:.5f} V")
    logger.info(f"cvRMSE : {metrics_rolling['cvRMSE']:.3f} %")
    logger.info(f"NMBE   : {metrics_rolling['NMBE']:.3f} %")
    logger.info(f"MDA    : {metrics_rolling['MDA']:.2f} %")
    logger.info(f"R2     : {metrics_rolling['R2']:.4f}\n")
    
    logger.info("--- Autoregressive Forecast (Blind Predict) ---")
    logger.info(f"MAE    : {metrics_ar['MAE']:.5f} V")
    logger.info(f"cvRMSE : {metrics_ar['cvRMSE']:.3f} %")
    logger.info(f"NMBE   : {metrics_ar['NMBE']:.3f} %")
    logger.info(f"MDA    : {metrics_ar['MDA']:.2f} %")
    logger.info(f"R2     : {metrics_ar['R2']:.4f}")
    logger.info("=========================================================\n")

    # 8. Save the Model
    logger.info(f"Saving trained TFT model to '{MODEL_SAVE_PATH}'...")
    tft_model.save(MODEL_SAVE_PATH)
    logger.info("Model saved successfully!")

    # 9. Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    test_dates = test_target.time_index
    
    ax.plot(test_dates, y_test_abs, label='Actual Electrode V', color='black', linewidth=2)
    ax.plot(test_dates, rolling_abs, label='Rolling Forecast (Frequent Sensor Updates)', color='#10b981', linestyle='--', linewidth=2)
    ax.plot(test_dates, ar_abs, label='Autoregressive Forecast (Long Horizon Blind Predict)', color='#ef4444', linestyle=':', linewidth=2)
    
    ax.set_title("TFT Model Evaluation - Absolute Electrode Potential", fontsize=15, fontweight='bold')
    ax.set_ylabel("Electrode Potential (V)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    output_img = "tft_evaluation_results.png"
    plt.savefig(output_img)
    logger.info(f"Plot saved as '{output_img}'. Check your directory!")
    plt.show()

if __name__ == "__main__":
    main()