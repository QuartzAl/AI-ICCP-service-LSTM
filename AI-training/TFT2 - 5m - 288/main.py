#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - TFT Hyperparameter Search Script
Searches for the optimal Temporal Fusion Transformer architecture using Optuna.
Applies First-Order Differencing for stationarity and rolling forecasts.
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

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Metrics and Scaling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from pytorch_lightning.callbacks import EarlyStopping

# Darts (PyTorch-based Time Series Library)
from darts import TimeSeries, concatenate
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
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

# --- Tunable Data & Prediction Parameters ---
INFLUX_AGG_WINDOW = "5m"       # InfluxDB aggregate Window (e.g., "2m", "10m", "1h")
PANDAS_FREQ = "5min"           # Pandas Resample Frequency (e.g., "2min", "10min", "1h")

PREDICTION_GAP = 288             # The Gap: How many steps to skip forward into the future before predicting

TRAIN_SPLIT = 0.80             # 80% train, 20% test 
RESULTS_FILE = "timeseries_models_metrics.txt"
PREDICTIONS_CSV = "timeseries_predictions.csv"
# =========================================================


def query_influxdb_data() -> pd.DataFrame:
    """Queries telemetry from InfluxDB and aggregates it."""
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
        
        if df.empty:
            raise ValueError("Query returned an empty dataset.")
            
        df = df.rename(columns={"_time": "time"})
        
        # Strip timezone and strict frequency indexing for Darts
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df.set_index('time', inplace=True)
        
        features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]
        return df[features]
    except Exception as e:
        logger.warning(f"Could not load live InfluxDB data: {e}")
        return None

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculates evaluation metrics for time series validation."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    mean_actual = np.mean(actual)
    
    cv_rmse = (rmse / mean_actual) * 100 if mean_actual != 0 else 0.0
    nmbe = (np.sum(actual - predicted) / (len(actual) * mean_actual)) * 100 if mean_actual != 0 else 0.0
    r2 = r2_score(actual, predicted)
    
    actual_diff = np.sign(actual[1:] - actual[:-1])
    pred_diff = np.sign(predicted[1:] - actual[:-1])
    mda = np.mean(actual_diff == pred_diff) * 100
    
    return {
        "MAE": mae,
        "cvRMSE": cv_rmse,
        "NMBE": nmbe,
        "MDA": mda,
        "R2": r2
    }

def reconstruct_absolute(pred_deltas, df_base, time_index, stride, target_col="electrode_V"):
    """
    Reconstructs absolute values from predicted deltas using a rolling base (stride).
    Uses the shifted dataframe base to align correctly with the delta.
    """
    pred_abs = []
    for i in range(0, len(pred_deltas), stride):
        chunk_deltas = pred_deltas[i : i+stride]
        current_time = time_index[i]
        
        # Get exact index in the base dataframe for current timestamp
        abs_idx = df_base.index.get_loc(current_time)
        
        # Base value is the actual absolute value immediately before this prediction chunk
        base_value = df_base[target_col].iloc[abs_idx - 1]
        
        # Reconstruct: Base + Cumulative Sum of Deltas
        chunk_abs = base_value + np.cumsum(chunk_deltas)
        pred_abs.extend(chunk_abs)
        
    return np.array(pred_abs)

def rolling_forecast(model, full_target_scaled, train_idx, test_length, past_cov, future_cov, chunk_out, stride):
    """
    Simulates a real-world production environment by predicting chunks, 
    skipping forward by the stride amount, grounding the model with actuals, and predicting again.
    """
    preds_list = []
    
    for i in range(0, test_length, stride):
        context_end_idx = train_idx + i
        
        # Boundary Check: TFT *must* have `chunk_out` future covariates available to satisfy 
        # its fixed decoder requirements. If we're too close to the end, break the loop gracefully.
        if context_end_idx + chunk_out > len(full_target_scaled):
            break
            
        current_context = full_target_scaled[:context_end_idx]
        
        # Predict the full chunk strictly to satisfy the TFT architecture constraints
        pred = model.predict(
            n=chunk_out,
            series=current_context,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False
        )
        
        # We only KEEP the number of steps defined by our stride to prevent overlapping timelines
        steps_to_keep = min(stride, test_length - i)
        preds_list.append(pred[:steps_to_keep])
        
    if not preds_list:
        raise ValueError("Test set is too small to perform even one rolling forecast chunk.")
        
    return concatenate(preds_list)

def main():
    df_abs = query_influxdb_data()
    if df_abs is None:
        logger.error("Could not get influxdb data. Exiting.")
        return
        
    df_abs = df_abs.asfreq(PANDAS_FREQ).ffill().bfill()
    df_abs = df_abs.dropna()
    
    target_col = "electrode_V"
    past_cov_cols = ["bus_voltage_V", "current_mA", "soil_humidity_V"]
    future_cov_cols = ["target_current_mA"] 
    
    # ---------------------------------------------------------
    # TARGET SHIFTING: CREATE THE PREDICTION GAP
    # ---------------------------------------------------------
    logger.info(f"Shifting target and future covariates backward by {PREDICTION_GAP} steps to create prediction gap...")
    df_shifted = df_abs.copy()
    
    # Shift the target so index 't' actually holds the target for 't + GAP'
    df_shifted[target_col] = df_shifted[target_col].shift(-PREDICTION_GAP)
    
    # Shift future covariates so they perfectly align with the new target timestamps
    for col in future_cov_cols:
        df_shifted[col] = df_shifted[col].shift(-PREDICTION_GAP)
        
    df_shifted = df_shifted.dropna()
    
    # Apply First-Order Differencing for Stationarity to the SHIFTED dataframe
    logger.info("Applying first-order differencing to all variables...")
    df_diff = df_shifted.diff().dropna()
    
    target_ts = TimeSeries.from_dataframe(df_diff, value_cols=target_col)
    past_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=past_cov_cols)
    future_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=future_cov_cols)
    
    train_idx = int(len(df_diff) * TRAIN_SPLIT)
    
    # Split Targets 
    train_target = target_ts[:train_idx]
    test_target = target_ts[train_idx:]
    
    # Split Covariates
    train_past_cov = past_cov_ts[:train_idx]
    test_past_cov = past_cov_ts[train_idx:]
    
    train_future_cov = future_cov_ts[:train_idx]
    test_future_cov = future_cov_ts[train_idx:]
    
    # Scaling (StandardScaler to prevent outlier squashing)
    logger.info("Scaling features and targets (StandardScaler)...")
    target_scaler = Scaler(StandardScaler())
    past_cov_scaler = Scaler(StandardScaler())
    future_cov_scaler = Scaler(StandardScaler())
    
    # Fit strictly on train to prevent data leakage
    target_scaler.fit(train_target)
    past_cov_scaler.fit(train_past_cov)
    future_cov_scaler.fit(train_future_cov)
    
    # Transform everything for context windows
    full_target_scaled = target_scaler.transform(target_ts)
    train_target_scaled = full_target_scaled[:train_idx]
    test_target_scaled = full_target_scaled[train_idx:]
    
    full_past_cov_scaled = past_cov_scaler.transform(past_cov_ts)
    train_past_cov_scaled = full_past_cov_scaled[:train_idx]
    test_past_cov_scaled = full_past_cov_scaled[train_idx:]
    
    full_future_cov_scaled = future_cov_scaler.transform(future_cov_ts)
    train_future_cov_scaled = full_future_cov_scaled[:train_idx]
    test_future_cov_scaled = full_future_cov_scaled[train_idx:]
    
    y_test_delta = test_target.values().flatten()
    test_length = len(y_test_delta)
    
    # Extract the ground truth absolute values for the exact test index timestamps
    y_test_abs = df_shifted.loc[test_target.time_index, target_col].values
    
    predictions_delta = {}
    predictions_abs = {}
    model_params = {}
    
    # ---------------------------------------------------------
    # Model: Temporal Fusion Transformer (TFT) with Optuna
    # ---------------------------------------------------------
    logger.info("Initializing Optuna Hyperparameter Optimization for TFT...")
    
    def objective(trial):
        in_chunk = trial.suggest_int("input_chunk_length", 30, 120, step=30)
        out_chunk = trial.suggest_int("output_chunk_length", 5, 60, step=5)
        hidden_size = trial.suggest_int("hidden_size", 16, 64, step=16)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        num_attention_heads = trial.suggest_categorical("num_attention_heads", [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            mode="min"
        )
        
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
            pl_trainer_kwargs={
                "devices": 1, 
                "callbacks": [early_stopper, pruner_callback],
                "enable_progress_bar": True
            }      
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
        
        # Evaluate Trial using the Rolling Forecast
        predictions_scaled = rolling_forecast(
            tft_model_trial, 
            full_target_scaled, 
            train_idx, 
            test_length, 
            full_past_cov_scaled, 
            full_future_cov_scaled, 
            chunk_out=out_chunk,
            stride=out_chunk 
        )
        
        # Ensure we only evaluate the ground truth steps that the rolling forecast was able to safely predict
        pred_len = len(predictions_scaled)
        preds_delta = target_scaler.inverse_transform(predictions_scaled).values().flatten()
        preds_abs = reconstruct_absolute(preds_delta, df_shifted, test_target.time_index[:pred_len], stride=out_chunk)
        
        # Evaluate MAE on Reconstructed Absolute Volts (sliced to match predicted length)
        mae = mean_absolute_error(y_test_abs[:pred_len], preds_abs)
        
        return mae

    try:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(
            direction="minimize", 
            study_name="TFT_Hyperparameter_Optimization",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3) 
        )
        
        logger.info("Starting Optuna search (10 trials)...")
        study.optimize(objective, n_trials=10)
        
        best_params = study.best_params
        logger.info(f"Optuna Search Complete! Best Trial MAE: {study.best_value:.5f} V")
        logger.info(f"Best Parameters: {best_params}")
        
        # ---------------------------------------------------------
        # Train FINAL Model using Optuna's Best Parameters
        # ---------------------------------------------------------
        logger.info("Training FINAL TFT model with best parameters...")
        tft_epochs = 150
        
        final_early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
            mode="min"
        )
        
        final_tft_model = TFTModel(
            input_chunk_length=best_params["input_chunk_length"],
            output_chunk_length=best_params["output_chunk_length"],
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
            pl_trainer_kwargs={
                "devices": 1, 
                "callbacks": [final_early_stopper]
            }      
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
        
        logger.info("Forecasting with FINAL TFT model (Rolling prediction)...")
        
        # Use Rolling Forecast for the final evaluation
        predictions_scaled = rolling_forecast(
            final_tft_model, 
            full_target_scaled, 
            train_idx, 
            test_length, 
            full_past_cov_scaled, 
            full_future_cov_scaled, 
            chunk_out=best_params["output_chunk_length"],
            stride=best_params["output_chunk_length"]
        )
        
        pred_len = len(predictions_scaled)
        preds_delta = target_scaler.inverse_transform(predictions_scaled).values().flatten()
        preds_abs = reconstruct_absolute(preds_delta, df_shifted, test_target.time_index[:pred_len], stride=best_params["output_chunk_length"])
        # def reconstruct_absolute(pred_deltas, df_base, time_index, stride, target_col="electrode_V"):
        
        predictions_delta['TFT'] = preds_delta
        predictions_abs['TFT'] = preds_abs
        
        model_params['TFT'] = {
            "input_chunk_length": best_params["input_chunk_length"],
            "output_chunk_length": best_params["output_chunk_length"],
            "prediction_gap": PREDICTION_GAP,
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
    # 5. Evaluate Metrics, Export to TXT and Save Predictions to CSV
    # ---------------------------------------------------------
    if not predictions_abs:
        logger.error("No predictions were generated. Check hardware constraints or data sizes.")
        return

    logger.info("Evaluating predictions and saving metrics...")
    
    # Ensure test matrices perfectly align with the safe predicted output bounds
    y_test_abs_sliced = y_test_abs[:pred_len]
    y_test_delta_sliced = y_test_delta[:pred_len]
    
    with open(RESULTS_FILE, "w") as f:
        f.write("=========================================================\n")
        f.write(f" TIME SERIES MODELS VALIDATION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=========================================================\n\n")
        
        for model_name, preds_abs in predictions_abs.items():
            metrics = calculate_metrics(y_test_abs_sliced, preds_abs)
            
            f.write(f"--- Model: {model_name} (Evaluated on Absolute Values) ---\n")
            f.write("Parameters:\n")
            for k, v in model_params[model_name].items():
                f.write(f"  - {k}: {v}\n")
                
            f.write("Metrics:\n")
            f.write(f"  * MAE    (Primary) : {metrics['MAE']:.5f} V\n")
            f.write(f"  * cvRMSE           : {metrics['cvRMSE']:.3f} %\n")
            f.write(f"  * NMBE             : {metrics['NMBE']:.3f} %\n")
            f.write(f"  * MDA              : {metrics['MDA']:.2f} %\n")
            f.write(f"  * R2 Score         : {metrics['R2']:.4f}\n\n")
            
    logger.info(f"Results successfully saved to '{RESULTS_FILE}'")

    # Shift plot dates to represent the actual future target times for CSV and Plots
    freq_offset = pd.to_timedelta(PANDAS_FREQ) * PREDICTION_GAP
    real_future_dates = test_target.time_index[:pred_len] + freq_offset

    results_df = pd.DataFrame({
        "Real_Future_Timestamp": real_future_dates,
        "Actual_Electrode_V_Abs": y_test_abs_sliced,
        "Actual_Electrode_V_Delta": y_test_delta_sliced
    })
    
    for model_name in predictions_abs.keys():
        results_df[f"{model_name}_Prediction_V_Abs"] = predictions_abs[model_name]
        results_df[f"{model_name}_Prediction_V_Delta"] = predictions_delta[model_name]
        
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    logger.info(f"Predictions successfully saved to '{PREDICTIONS_CSV}'")

    # ---------------------------------------------------------
    # 6. Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
    colors = {'TFT': '#8b5cf6'}
    
    # --- Plot 1: Absolute Values ---
    axes[0].plot(real_future_dates, y_test_abs_sliced, label='Actual Electrode V (Absolute)', color='black', linewidth=2)
    for model_name, preds_abs in predictions_abs.items():
        axes[0].plot(real_future_dates, preds_abs, label=f"{model_name} Forecast (Absolute)", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    axes[0].set_title(f"Deep Learning Forecast Comparison (Gap: {PREDICTION_GAP} Steps) - Absolute Values", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Electrode Potential (V)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Delta (Differenced) Values ---
    axes[1].plot(real_future_dates, y_test_delta_sliced, label='Actual Δ Electrode V (Delta)', color='gray', linewidth=2)
    for model_name, preds_delta in predictions_delta.items():
        axes[1].plot(real_future_dates, preds_delta, label=f"{model_name} Forecast (Delta)", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    axes[1].set_title("Deep Learning Forecast Comparison - First-Order Differenced (Delta)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Δ Electrode Potential (V)")
    axes[1].set_xlabel("Real Future Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_img = "timeseries_forecast_results_differenced.png"
    plt.savefig(output_img)
    logger.info(f"Analysis chart saved as: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()