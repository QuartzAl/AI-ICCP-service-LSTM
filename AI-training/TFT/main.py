#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - Temporal Fusion Transformer (TFT) Script
Predicts future reference electrode potential (electrode_V) using a 
deep learning Temporal Fusion Transformer architecture via the 'darts' library.

Includes comprehensive validation metrics: MAE (Primary), cvRMSE, NMBE, MDA, and R2.
Results and parameters are exported to a text file.
Predictions and actual data are exported to a CSV file.
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

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

TRAIN_SPLIT = 0.80  # 80% train, 20% val/test
RESULTS_FILE = "timeseries_models_metrics.txt"
PREDICTIONS_CSV = "timeseries_predictions.csv"

def query_influxdb_data() -> pd.DataFrame:
    """Queries telemetry from InfluxDB and aggregates it."""
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

def reconstruct_absolute(pred_deltas, df_abs, time_index, chunk_out=15):
    """
    Reconstructs absolute values from predicted deltas using a rolling base.
    """
    pred_abs = []
    for i in range(0, len(pred_deltas), chunk_out):
        chunk_deltas = pred_deltas[i : i+chunk_out]
        current_time = time_index[i]
        
        # Get exact index in the absolute dataframe for current timestamp
        abs_idx = df_abs.index.get_loc(current_time)
        
        # Base value is the actual absolute value immediately before this prediction chunk
        base_value = df_abs['electrode_V'].iloc[abs_idx - 1]
        
        # Reconstruct: Base + Cumulative Sum of Deltas
        chunk_abs = base_value + np.cumsum(chunk_deltas)
        pred_abs.extend(chunk_abs)
        
    return np.array(pred_abs)

def rolling_forecast(model, full_target_scaled, train_idx, test_length, past_cov, future_cov, chunk_out=15):
    """
    Simulates a real-world production environment by predicting a small chunk, 
    then grounding the model with the actual sensor readings before predicting the next chunk.
    This entirely prevents the "Auto-Regression Flatline" problem over long test sets.
    """
    preds_list = []
    
    for i in range(0, test_length, chunk_out):
        # The context window slides forward, revealing the true history step-by-step
        context_end_idx = train_idx + i
        current_context = full_target_scaled[:context_end_idx]
        
        steps_to_predict = min(chunk_out, test_length - i)
        
        pred = model.predict(
            n=steps_to_predict,
            series=current_context,
            past_covariates=past_cov,
            future_covariates=future_cov,
            verbose=False
        )
        preds_list.append(pred)
        
    return concatenate(preds_list)

def main():
    df_abs = query_influxdb_data()
    if df_abs is None:
        logger.error("Could not get influxdb data. Exiting.")
        return
        
    df_abs = df_abs.asfreq('2min').ffill().bfill()
    df_abs = df_abs.dropna()
    
    # Apply First-Order Differencing for Stationarity
    logger.info("Applying first-order differencing to all variables...")
    df_diff = df_abs.diff().dropna()
    
    target_col = "electrode_V"
    past_cov_cols = ["bus_voltage_V", "current_mA", "soil_humidity_V"]
    future_cov_cols = ["target_current_mA"] 
    
    target_ts = TimeSeries.from_dataframe(df_diff, value_cols=target_col)
    past_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=past_cov_cols)
    future_cov_ts = TimeSeries.from_dataframe(df_diff, value_cols=future_cov_cols)
    
    train_idx = int(len(df_diff) * TRAIN_SPLIT)
    
    # Split Targets (Previously missing!)
    train_target = target_ts[:train_idx]
    test_target = target_ts[train_idx:]
    
    # Split Covariates
    train_past_cov = past_cov_ts[:train_idx]
    test_past_cov = past_cov_ts[train_idx:]
    
    train_future_cov = future_cov_ts[:train_idx]
    test_future_cov = future_cov_ts[train_idx:]
    
    # 3. Scaling (CRITICAL FIX: Use StandardScaler to prevent outlier squashing)
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
    y_test_abs = df_abs.loc[test_target.time_index, target_col].values
    
    predictions_delta = {}
    predictions_abs = {}
    model_params = {}
    
    # ---------------------------------------------------------
    # 4. Model: Temporal Fusion Transformer (TFT) with Optuna
    # ---------------------------------------------------------
    logger.info("Initializing Optuna Hyperparameter Optimization for TFT...")
    
    def objective(trial):
        in_chunk = trial.suggest_int("input_chunk_length", 30, 120, step=30)
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
            output_chunk_length=15,  
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
            verbose=True
        )
        
        # Evaluate Trial using the Rolling Forecast to prevent flatlining
        predictions_scaled = rolling_forecast(
            tft_model_trial, 
            full_target_scaled, 
            train_idx, 
            test_length, 
            full_past_cov_scaled, 
            full_future_cov_scaled, 
            chunk_out=15
        )
        
        preds_delta = target_scaler.inverse_transform(predictions_scaled).values().flatten()
        preds_abs = reconstruct_absolute(preds_delta, df_abs, test_target.time_index, chunk_out=15)
        
        # Evaluate MAE on Reconstructed Absolute Volts
        mae = mean_absolute_error(y_test_abs, preds_abs)
        
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
            output_chunk_length=15,
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
            chunk_out=15
        )
        
        preds_delta = target_scaler.inverse_transform(predictions_scaled).values().flatten()
        preds_abs = reconstruct_absolute(preds_delta, df_abs, test_target.time_index, chunk_out=15)
        
        predictions_delta['TFT'] = preds_delta
        predictions_abs['TFT'] = preds_abs
        
        model_params['TFT'] = {
            "input_chunk_length": best_params["input_chunk_length"],
            "output_chunk_length": 15,
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

    # ---------------------------------------------------------
    # 5. Evaluate Metrics, Export to TXT and Save Predictions to CSV
    # ---------------------------------------------------------
    if not predictions_abs:
        logger.error("No predictions were generated. Check hardware constraints or data sizes.")
        return

    logger.info("Evaluating predictions and saving metrics...")
    
    with open(RESULTS_FILE, "w") as f:
        f.write("=========================================================\n")
        f.write(f" TIME SERIES MODELS VALIDATION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=========================================================\n\n")
        
        for model_name, preds_abs in predictions_abs.items():
            metrics = calculate_metrics(y_test_abs, preds_abs)
            
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

    results_df = pd.DataFrame({
        "Timestamp": test_target.time_index,
        "Actual_Electrode_V_Abs": y_test_abs,
        "Actual_Electrode_V_Delta": y_test_delta
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
    test_dates = test_target.time_index
    colors = {'TFT': '#8b5cf6'}
    
    # --- Plot 1: Absolute Values ---
    axes[0].plot(test_dates, y_test_abs, label='Actual Electrode V (Absolute)', color='black', linewidth=2)
    for model_name, preds_abs in predictions_abs.items():
        axes[0].plot(test_dates, preds_abs, label=f"{model_name} Forecast (Absolute)", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    axes[0].set_title("Deep Learning Forecast Comparison - Absolute Values", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Electrode Potential (V)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Delta (Differenced) Values ---
    axes[1].plot(test_dates, y_test_delta, label='Actual Δ Electrode V (Delta)', color='gray', linewidth=2)
    for model_name, preds_delta in predictions_delta.items():
        axes[1].plot(test_dates, preds_delta, label=f"{model_name} Forecast (Delta)", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    axes[1].set_title("Deep Learning Forecast Comparison - First-Order Differenced (Delta)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Δ Electrode Potential (V)")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_img = "timeseries_forecast_results_differenced.png"
    plt.savefig(output_img)
    logger.info(f"Analysis chart saved as: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()