#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - Statistical Time Series Script
Predicts future reference electrode potential (electrode_V) using 
Exponential Smoothing (ETS), Vector Autoregression (VAR), and 
Structural Time Series (STS) models.

Includes comprehensive validation metrics: MAE (Primary), cvRMSE, NMBE, MDA, and R2.
Results and parameters are exported to a text file.
Predictions and actual data are exported to a CSV file.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Statistical Time Series Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Suppress harmless statsmodels warnings for index frequency
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-TimeSeries")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

TRAIN_SPLIT = 0.8
RESULTS_FILE = "timeseries_models_metrics.txt"
PREDICTIONS_CSV = "timeseries_predictions.csv"

def query_influxdb_data() -> pd.DataFrame:
    """Queries telemetry from InfluxDB and aggregates it."""
    logger.info(f"Querying InfluxDB at {INFLUXDB_URL} for device_id '{DEVICE_ID}'...")
    
    flux_query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -25d)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 30m, fn: mean, createEmpty: true)
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

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculates evaluation metrics for time series validation."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    mean_actual = np.mean(actual)
    
    # Coefficient of Variation of RMSE (%)
    cv_rmse = (rmse / mean_actual) * 100 if mean_actual != 0 else 0.0
    
    # Normalized Mean Bias Error (%)
    nmbe = (np.sum(actual - predicted) / (len(actual) * mean_actual)) * 100 if mean_actual != 0 else 0.0
    
    # R2 Score
    r2 = r2_score(actual, predicted)
    
    # Mean Directional Accuracy (MDA) (%)
    # Compares the sign of the actual difference to the predicted difference
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

def main():
    # 1. Fetch and Prepare Data
    df = query_influxdb_data()
    if df is None:
        logger.error("Could not get influxdb data. Exiting.")
        return
        
    df = df.ffill().bfill()
    df = df.dropna()
    
    # Train/Test Split (Chronological)
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    target_col = "electrode_V"
    target_idx = df.columns.get_loc(target_col)
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    forecast_steps = len(y_test)
    
    predictions = {}
    model_params = {}
    
    # ---------------------------------------------------------
    # 2. Model: Exponential Smoothing (ETS) - Univariate
    # ---------------------------------------------------------
    logger.info("Fitting Exponential Smoothing (ETS) model...")
    try:
        ets_model = ExponentialSmoothing(y_train, trend='add', seasonal=None, initialization_method="estimated")
        ets_fit = ets_model.fit()
        predictions['ETS'] = ets_fit.forecast(forecast_steps)
        model_params['ETS'] = {
            "smoothing_level": ets_fit.params['smoothing_level'],
            "smoothing_trend": ets_fit.params['smoothing_trend']
        }
    except Exception as e:
        logger.error(f"ETS failed: {e}")

    # ---------------------------------------------------------
    # 3. Model: Vector Autoregression (VAR) - Multivariate
    # ---------------------------------------------------------
    logger.info("Fitting Vector Autoregression (VAR) model...")
    try:
        var_model = VAR(train_df.values)
        # Select optimal lag order based on AIC
        var_results = var_model.fit(maxlags=15, ic='aic')
        lag_order = var_results.k_ar
        
        # Forecast requires the last 'lag_order' observations from train data
        forecast_input = train_df.values[-lag_order:]
        var_forecast = var_results.forecast(y=forecast_input, steps=forecast_steps)
        
        # Extract the target variable column
        predictions['VAR'] = var_forecast[:, target_idx]
        model_params['VAR'] = {"optimal_lag_order_aic": lag_order}
    except Exception as e:
        logger.error(f"VAR failed: {e}")

    # ---------------------------------------------------------
    # 4. Model: Structural Time Series (STS) - Univariate
    # ---------------------------------------------------------
    logger.info("Fitting Structural Time Series (STS) model...")
    try:
        # Local Linear Trend model setup
        sts_model = UnobservedComponents(y_train, level='local linear trend')
        sts_fit = sts_model.fit(disp=False)
        predictions['STS'] = sts_fit.forecast(steps=forecast_steps)
        model_params['STS'] = {
            "sigma2.measurement": sts_fit.params[0],
            "sigma2.level": sts_fit.params[1],
            "sigma2.trend": sts_fit.params[2]
        }
    except Exception as e:
        logger.error(f"STS failed: {e}")

    # ---------------------------------------------------------
    # 5. Evaluate Metrics, Export to TXT and Save Predictions to CSV
    # ---------------------------------------------------------
    logger.info("Evaluating predictions and saving metrics...")
    
    # Save Metrics to TXT
    with open(RESULTS_FILE, "w") as f:
        f.write("=========================================================\n")
        f.write(f" TIME SERIES MODELS VALIDATION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=========================================================\n\n")
        
        for model_name, preds in predictions.items():
            metrics = calculate_metrics(y_test, preds)
            
            f.write(f"--- Model: {model_name} ---\n")
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

    # Save Predictions to CSV
    logger.info("Generating predictions CSV...")
    results_df = pd.DataFrame({
        "Timestamp": test_df.index,
        "Actual_Electrode_V": y_test
    })
    
    # Add each successful model's prediction as a new column
    for model_name, preds in predictions.items():
        results_df[f"{model_name}_Prediction_V"] = preds
        
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    logger.info(f"Predictions successfully saved to '{PREDICTIONS_CSV}'")

    # ---------------------------------------------------------
    # 6. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 7))
    test_dates = test_df.index
    
    plt.plot(test_dates, y_test, label='Actual Electrode V', color='black', linewidth=2)
    
    colors = {'ETS': '#3b82f6', 'VAR': '#ef4444', 'STS': '#10b981'}
    for model_name, preds in predictions.items():
        plt.plot(test_dates, preds, label=f"{model_name} Forecast", color=colors.get(model_name), linestyle='--', alpha=0.8)
        
    plt.title("Statistical Time Series Forecast Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Electrode Potential (V)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_img = "timeseries_forecast_results.png"
    plt.savefig(output_img)
    logger.info(f"Analysis chart saved as: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()