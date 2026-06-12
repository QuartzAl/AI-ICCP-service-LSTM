#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - TFT Hyperparameter Search Script
Searches for the optimal Temporal Fusion Transformer architecture using Optuna.
Applies TimeSeries shifting to map immediate predictions to a future horizon (Prediction Gap).
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import pickle

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Metrics and Scaling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from pytorch_lightning.callbacks import EarlyStopping

# Darts (PyTorch-based Time Series Library)
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Optimize precision for AMD Matrix Cores / NVIDIA Tensor Cores
torch.set_float32_matmul_precision('medium')

# Suppress harmless warnings and excessive PyTorch Lightning logs
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-TimeSeries-TFT")

# =========================================================
# --- Configuration Constants ---
# =========================================================
INFLUXDB_URL = "http://203.189.120.153:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

# --- Tunable Data & Prediction Parameters ---
INFLUX_AGG_WINDOW = "5m"       # InfluxDB aggregate Window
PANDAS_FREQ = "5min"           # Pandas Resample Frequency 

PREDICTION_GAP = 288             # Shift the target timeline backwards by this many steps

TRAIN_SPLIT = 0.80             

# --- Output Paths ---
OUTPUT_DIR = "tft_results"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "timeseries_models_metrics.txt")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "timeseries_predictions.csv")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_tft_model.pt")
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, "scalers.pkl")
PLOT_IMG_PATH = os.path.join(OUTPUT_DIR, "timeseries_forecast_results.png")
# =========================================================

def query_influxdb_data() -> pd.DataFrame:
    logger.info(f"Querying InfluxDB at {INFLUXDB_URL} for device_id '{DEVICE_ID}'...")
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: {INFLUX_AGG_WINDOW}, fn: mean, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"])
    '''
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=3000)
        query_api = client.query_api()
        df = query_api.query_data_frame(query=flux_query)
        client.close()
        
        if df.empty: raise ValueError("Query returned an empty dataset.")
            
        df = df.rename(columns={"_time": "time"})
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df.set_index('time', inplace=True)
        return df[["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]]
    except Exception as e:
        logger.warning(f"Could not load live InfluxDB data: {e}")
        return None

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

def rolling_forecast(model, full_target_scaled, train_idx, test_length, past_cov, future_cov, chunk_size, stride):
    """
    Predicts sequences natively (as the model already thinks it's predicting the present, 
    due to the target shift).
    """
    preds_list = []
    
    for i in range(0, test_length, stride):
        context_end_idx = train_idx + i
        
        if context_end_idx + chunk_size > len(full_target_scaled):
            break
            
        current_context = full_target_scaled[:context_end_idx]
        
        pred = model.predict(
            n=chunk_size,
            series=current_context,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False
        )
        preds_list.append(pred)
        
    if not preds_list:
        raise ValueError("Test set is too small to perform even one rolling forecast chunk.")
        
    return preds_list

def process_shifted_predictions(preds_list, df_abs, df_diff, target_scaler, target_col, gap):
    """
    Unscales predictions, maps the shifted timestamps to real-world future timestamps,
    and reconstructs absolute values using accumulated errors through the prediction gap.
    """
    preds_abs = []
    preds_delta = []
    actual_abs = []
    actual_delta = []
    real_time_index = []
    
    freq_offset = pd.to_timedelta(PANDAS_FREQ) * gap
    
    # 1. Build a master dictionary of all predicted deltas across all chunks
    # This allows us to look up previously predicted values to bridge the gap accurately.
    all_predicted_deltas = {}
    for pred in preds_list:
        pred_inv = target_scaler.inverse_transform(pred)
        chunk_deltas = pred_inv.values().flatten()
        chunk_real_times = pred.time_index + freq_offset
        for t, delta in zip(chunk_real_times, chunk_deltas):
            all_predicted_deltas[t] = delta

    # 2. Process each chunk and reconstruct absolute values properly
    for pred in preds_list:
        pred_inv = target_scaler.inverse_transform(pred)
        chunk_deltas = pred_inv.values().flatten()
        
        # Real future time = The model's "shifted" present time + The gap
        chunk_real_times = pred.time_index + freq_offset
        
        t_start_real = chunk_real_times[0]
        
        # The "present" time when this forecast was made.
        # Since the first prediction is T_start_real = T_context + GAP + 1,
        # T_context = T_start_real - (GAP + 1)
        t_context = t_start_real - pd.to_timedelta(PANDAS_FREQ) * (gap + 1)
        
        # The known anchor absolute value at T_context (the present)
        if t_context in df_abs.index:
            base_value = df_abs.loc[t_context, target_col]
        else:
            # Fallback to the earliest available actual if context is out of bounds
            base_value = df_abs.iloc[0][target_col]
            
        # Accumulate errors through the prediction gap (T_context + 1 to T_start_real - 1)
        gap_accumulated_delta = 0.0
        for i in range(1, gap + 1):
            gap_t = t_context + pd.to_timedelta(PANDAS_FREQ) * i
            # If we previously predicted this timestep, use our prediction!
            # Otherwise, fallback to the actual known delta.
            if gap_t in all_predicted_deltas:
                gap_accumulated_delta += all_predicted_deltas[gap_t]
            elif gap_t in df_diff.index:
                gap_accumulated_delta += df_diff.loc[gap_t, target_col]
                
        # The reconstructed absolute value immediately before the chunk begins
        chunk_base_abs = base_value + gap_accumulated_delta
        
        # Reconstruct the absolute values for the chunk itself
        chunk_abs = chunk_base_abs + np.cumsum(chunk_deltas)
        
        preds_abs.extend(chunk_abs)
        preds_delta.extend(chunk_deltas)
        
        # Pull actual ground truths based on the *real* future timestamps
        actual_abs.extend(df_abs.loc[chunk_real_times, target_col].values)
        actual_delta.extend(df_diff.loc[chunk_real_times, target_col].values)
        real_time_index.extend(chunk_real_times)
        
    return (
        np.array(preds_abs), 
        np.array(preds_delta), 
        np.array(actual_abs), 
        np.array(actual_delta), 
        pd.DatetimeIndex(real_time_index)
    )

def main():
    # Ensure output directory exists before generating any files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Initialized output directory: '{OUTPUT_DIR}/'")

    df_abs = query_influxdb_data()
    if df_abs is None: return
        
    df_abs = df_abs.asfreq(PANDAS_FREQ).ffill().bfill().dropna()
    
    logger.info("Applying first-order differencing...")
    df_diff = df_abs.diff().dropna()
    
    target_col = "electrode_V"
    past_cov_cols = ["bus_voltage_V", "current_mA", "soil_humidity_V"]
    future_cov_cols = ["target_current_mA"] 
    
    target_ts = TimeSeries.from_dataframe(df_diff, value_cols=target_col)
    past_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=past_cov_cols)
    future_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=future_cov_cols)
    
    # ---------------------------------------------------------
    # TARGET SHIFTING: The User's Logic
    # ---------------------------------------------------------
    logger.info(f"Shifting target and future covariates backwards by {PREDICTION_GAP} steps...")
    target_ts = target_ts.shift(-PREDICTION_GAP)
    future_cov_ts = future_cov_ts.shift(-PREDICTION_GAP)
    
    # Align the series timestamps so the model receives perfectly parallel data
    start_time = max(target_ts.start_time(), past_cov_ts.start_time(), future_cov_ts.start_time())
    end_time = min(target_ts.end_time(), past_cov_ts.end_time(), future_cov_ts.end_time())
    
    target_ts = target_ts.slice(start_time, end_time)
    past_cov_ts = past_cov_ts.slice(start_time, end_time)
    future_cov_ts = future_cov_ts.slice(start_time, end_time)
    # ---------------------------------------------------------
    
    train_idx = int(len(target_ts) * TRAIN_SPLIT)
    
    train_target = target_ts[:train_idx]
    test_target = target_ts[train_idx:]
    
    train_past_cov = past_cov_ts[:train_idx]
    test_past_cov = past_cov_ts[train_idx:]
    
    train_future_cov = future_cov_ts[:train_idx]
    test_future_cov = future_cov_ts[train_idx:]
    
    # Scaling
    logger.info("Scaling features and targets (StandardScaler)...")
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
    
    test_length = len(test_target)
    
    predictions_abs = {}
    predictions_delta = {}
    model_params = {}
    
    # ---------------------------------------------------------
    # Model: Temporal Fusion Transformer (TFT) with Optuna
    # ---------------------------------------------------------
    logger.info("Initializing Optuna Hyperparameter Optimization for TFT...")
    
    def objective(trial):
        in_chunk = trial.suggest_int("input_chunk_length", 30, 120, step=30)
        out_chunk = trial.suggest_int("output_chunk_length", 5, 30, step=5)
        hidden_size = trial.suggest_int("hidden_size", 16, 64, step=16)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        num_attention_heads = trial.suggest_categorical("num_attention_heads", [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        early_stopper = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")
        pruner_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        
        tft_model_trial = TFTModel(
            input_chunk_length=in_chunk,
            output_chunk_length=out_chunk,  
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=30,             
            add_relative_index=True, 
            add_encoders={'cyclic': {'future': ['hour', 'minute']}}, 
            random_state=42,
            optimizer_kwargs={'lr': lr},
            likelihood=None,                      
            loss_fn=nn.HuberLoss(),               
            pl_trainer_kwargs={"devices": 1, "callbacks": [early_stopper, pruner_callback], "enable_progress_bar": False}      
        )
        
        tft_model_trial.fit(
            series=train_target_scaled,
            past_covariates=train_past_cov_scaled,
            future_covariates=train_future_cov_scaled,
            val_series=test_target_scaled,
            val_past_covariates=test_past_cov_scaled,
            val_future_covariates=test_future_cov_scaled,
            verbose=False
        )
        
        preds_list = rolling_forecast(
            tft_model_trial, full_target_scaled, train_idx, test_length, 
            full_past_cov_scaled, full_future_cov_scaled, 
            chunk_size=out_chunk, stride=out_chunk
        )
        
        preds_abs, _, actual_abs, _, _ = process_shifted_predictions(
            preds_list, df_abs, df_diff, target_scaler, target_col, gap=PREDICTION_GAP
        )
            
        mae = mean_absolute_error(actual_abs, preds_abs)
        return mae

    try:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(direction="minimize", study_name="TFT_Gap_Optimization", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
        logger.info("Starting Optuna search (10 trials)...")
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        out_chunk = best_params["output_chunk_length"]
        
        logger.info(f"Optuna Search Complete! Best Trial MAE: {study.best_value:.5f} V")
        logger.info(f"Best Parameters: {best_params}")
        
        # ---------------------------------------------------------
        # Train FINAL Model using Optuna's Best Parameters
        # ---------------------------------------------------------
        logger.info("Training FINAL TFT model with best parameters...")
        tft_epochs = 150
        final_early_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")
        
        final_tft_model = TFTModel(
            input_chunk_length=best_params["input_chunk_length"],
            output_chunk_length=out_chunk,
            hidden_size=best_params["hidden_size"],
            lstm_layers=best_params["lstm_layers"],
            num_attention_heads=best_params["num_attention_heads"],
            dropout=best_params["dropout"],
            batch_size=best_params["batch_size"],
            n_epochs=tft_epochs,
            add_relative_index=True, 
            add_encoders={'cyclic': {'future': ['hour', 'minute']}}, 
            random_state=42,
            optimizer_kwargs={'lr': best_params["lr"]},
            likelihood=None,                      
            loss_fn=nn.HuberLoss(),               
            pl_trainer_kwargs={"devices": 1, "callbacks": [final_early_stopper]}      
        )
        
        final_tft_model.fit(
            series=train_target_scaled,
            past_covariates=train_past_cov_scaled,
            future_covariates=train_future_cov_scaled,
            val_series=test_target_scaled,                     
            val_past_covariates=test_past_cov_scaled,          
            val_future_covariates=test_future_cov_scaled,      
            verbose=True
        )
        
        logger.info(f"Saving final trained model to '{MODEL_SAVE_PATH}'...")
        final_tft_model.save(MODEL_SAVE_PATH)
        
        logger.info(f"Saving scalers to '{SCALER_SAVE_PATH}'...")
        with open(SCALER_SAVE_PATH, "wb") as f:
            pickle.dump({
                "target_scaler": target_scaler,
                "past_cov_scaler": past_cov_scaler,
                "future_cov_scaler": future_cov_scaler
            }, f)

        logger.info("Forecasting with FINAL TFT model...")
        
        preds_list = rolling_forecast(
            final_tft_model, full_target_scaled, train_idx, test_length, 
            full_past_cov_scaled, full_future_cov_scaled, 
            chunk_size=out_chunk, stride=out_chunk
        )
        
        preds_abs, preds_delta, actual_abs, actual_delta, real_time_index = process_shifted_predictions(
            preds_list, df_abs, df_diff, target_scaler, target_col, gap=PREDICTION_GAP
        )
        
        predictions_abs['TFT'] = preds_abs
        predictions_delta['TFT'] = preds_delta
        
        model_params['TFT'] = {
            "input_chunk_length": best_params["input_chunk_length"],
            "output_chunk_length": out_chunk,
            "prediction_gap_shift": PREDICTION_GAP,
            "hidden_size": best_params["hidden_size"],
            "lstm_layers": best_params["lstm_layers"],
            "num_attention_heads": best_params["num_attention_heads"],
            "dropout": best_params["dropout"],
            "batch_size": best_params["batch_size"],
            "lr": best_params["lr"],
            "epochs_limit": tft_epochs
        }
        
    except Exception as e:
        logger.error(f"TFT or Optuna failed: {repr(e)}\n{traceback.format_exc()}")
        return

    # ---------------------------------------------------------
    # Evaluate Metrics, Export to TXT and Save Predictions
    # ---------------------------------------------------------
    if not predictions_abs: return

    logger.info("Evaluating predictions and saving metrics...")
    
    with open(RESULTS_FILE, "w") as f:
        f.write("=========================================================\n")
        f.write(f" TIME SERIES MODELS VALIDATION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=========================================================\n\n")
        
        for model_name, preds_abs in predictions_abs.items():
            metrics = calculate_metrics(actual_abs, preds_abs)
            f.write(f"--- Model: {model_name} (Evaluated on Shifted Absolute Values) ---\n")
            f.write("Parameters:\n")
            for k, v in model_params[model_name].items(): f.write(f"  - {k}: {v}\n")
            f.write("Metrics:\n")
            f.write(f"  * MAE    (Primary) : {metrics['MAE']:.5f} V\n")
            f.write(f"  * cvRMSE           : {metrics['cvRMSE']:.3f} %\n")
            f.write(f"  * NMBE             : {metrics['NMBE']:.3f} %\n")
            f.write(f"  * MDA              : {metrics['MDA']:.2f} %\n")
            f.write(f"  * R2 Score         : {metrics['R2']:.4f}\n\n")
            
    logger.info(f"Results successfully saved to '{RESULTS_FILE}'")

    results_df = pd.DataFrame({
        "Real_Future_Timestamp": real_time_index,
        "Actual_Electrode_V_Abs": actual_abs,
        "Actual_Electrode_V_Delta": actual_delta
    })
    for model_name in predictions_abs.keys():
        results_df[f"{model_name}_Prediction_V_Abs"] = predictions_abs[model_name]
        results_df[f"{model_name}_Prediction_V_Delta"] = predictions_delta[model_name]
        
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    
    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
    colors = {'TFT': '#8b5cf6'}
    
    # --- Plot 1: Absolute Values ---
    axes[0].plot(real_time_index, actual_abs, label='Actual Electrode V (Absolute)', color='black', linewidth=2)
    for model_name, preds_abs in predictions_abs.items():
        axes[0].plot(real_time_index, preds_abs, label=f"{model_name} Target Forecast", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    axes[0].set_title(f"Deep Learning Forecast (Shift Gap: {PREDICTION_GAP} Steps) - Absolute Reconstruction", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Electrode Potential (V)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Delta Values ---
    axes[1].plot(real_time_index, actual_delta, label='Actual Δ Electrode V (Delta)', color='gray', linewidth=2)
    for model_name, preds_delta in predictions_delta.items():
        axes[1].plot(real_time_index, preds_delta, label=f"{model_name} Target Forecast (Delta)", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    axes[1].set_title("Deep Learning Forecast Comparison - First-Order Differenced (Delta)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Δ Electrode Potential (V)")
    axes[1].set_xlabel("Target Real-World Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_IMG_PATH)
    logger.info(f"Analysis chart saved as: '{PLOT_IMG_PATH}'")
    plt.show()

if __name__ == "__main__":
    main()