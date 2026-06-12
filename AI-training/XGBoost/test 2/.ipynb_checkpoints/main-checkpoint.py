#!/usr/bin/env python3
"""
ICCP Cathodic Protection Telemetry - XGBoost Predictive Modeling Script
Predicts future reference electrode potential (electrode_V) based on historical sequence data
combined with the context of future planned target currents.

Includes RandomizedSearchCV for hyperparameter tuning using TimeSeriesSplit 
and comprehensive evaluation metrics (MAE, RMSE, cvRMSE, NMBE, MDA, R2 Score).
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# InfluxDB Client library
from influxdb_client import InfluxDBClient

# Machine Learning framework libraries
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ICCP-XGB-FutureContext")

# --- Configuration Constants ---
INFLUXDB_URL = "http://edx.petra.ac.id:8086"
INFLUXDB_TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q=="
INFLUXDB_ORG = "petra"
INFLUXDB_BUCKET = "ICCP"
DEVICE_ID = "001"

LOOKBACK_WINDOW = 3
PREDICTION_STEP = 1
TRAIN_SPLIT = 0.8
SEARCH_ITERATIONS = 20  # Number of parameter settings sampled
RESULTS_FILE = "xgboost_evaluation_results.txt"


# --- Custom Metrics Functions ---
def calc_cvrmse(rmse, actual_data):
    """Coefficient of Variation of the RMSE (cvRMSE) in percentage."""
    mean_actual = np.mean(actual_data)
    if mean_actual == 0:
        return 0.0
    return (rmse / mean_actual) * 100.0

def calc_nmbe(actual_data, predicted_data):
    """Normalized Mean Bias Error (NMBE) in percentage."""
    mean_actual = np.mean(actual_data)
    if mean_actual == 0:
        return 0.0
    return (np.sum(actual_data - predicted_data) / (len(actual_data) * mean_actual)) * 100.0

def calc_mda(actual_data, predicted_data):
    """Mean Directional Accuracy (MDA) - Measures if model predicts the correct direction of change."""
    if len(actual_data) < 2:
        return 0.0
    actual_diff = np.sign(actual_data[1:] - actual_data[:-1])
    pred_diff = np.sign(predicted_data[1:] - actual_data[:-1])
    return np.mean(actual_diff == pred_diff)


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
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]
        return df[features]
    except Exception as e:
        logger.warning(f"Could not load live InfluxDB data: {e}")
        return None

def create_sequences_xgb(data: np.ndarray, lookback: int, pred_step: int):
    """
    Transforms raw arrays into 2D structural inputs for XGBoost.
    Instead of [samples, lookback, features], XGBoost needs [samples, lookback * features].
    """
    X, y = [], []
    for i in range(len(data) - lookback - pred_step + 1):
        # Flatten the 2D lookback window into a 1D array for this sample
        X.append(data[i:(i + lookback)].flatten())
        y.append(data[i + lookback + pred_step - 1, 2]) # 2 is index of electrode_V
    return np.array(X), np.array(y)


def main():
    # 1. Pipeline execution and cleaning
    df = query_influxdb_data()
    if df is None:
        logger.error("Could not get influxdb data. Exiting.")
        return
        
    df = df.ffill().bfill()
    df['future_target_current_mA'] = df['target_current_mA'].shift(-PREDICTION_STEP)
    df = df.dropna()
    
    features = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "future_target_current_mA"]
    data_values = df[features].values 
    num_features = data_values.shape[1]
    
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data_values)

    # Use the modified function to get flattened 2D data for XGBoost
    X, y = create_sequences_xgb(scaled_data, LOOKBACK_WINDOW, PREDICTION_STEP)
    print(X.shape)
    print(y.shape)
    logger.info(f"Generated XGBoost input tensor dimensions: X: {X.shape}, y: {y.shape}")

    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 2. Configure the automated Hyperparameter Optimization Tuner
    logger.info("Initializing Hyperparameter Tuning for XGBoost...")
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # TimeSeriesSplit prevents data leakage by maintaining temporal order during CV folds
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Setting MAE as the primary scoring metric for cross-validation evaluation
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring='neg_mean_absolute_error', # MAE is the main metric
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    logger.info("Searching for best parameters using Time Series Cross-Validation...")
    search.fit(X_train, y_train)
    
    best_hps = search.best_params_
    best_model = search.best_estimator_

    logger.info(f"--- OPTIMIZATION FOUND BEST PARAMETERS ---")
    for key, val in best_hps.items():
        logger.info(f"Best {key}: {val}")
        
    # 3. Model Evaluation & Metric Calculation
    logger.info("Evaluating final XGBoost model on test data...")
    predictions_scaled = best_model.predict(X_test)
    
    # Invert predictions back to scale
    temp_pred_matrix = np.zeros((len(predictions_scaled), num_features))
    temp_pred_matrix[:, 2] = predictions_scaled 
    predictions_real = scaler.inverse_transform(temp_pred_matrix)[:, 2]
    
    # Invert target actuals back to scale
    temp_actual_matrix = np.zeros((len(y_test), num_features))
    temp_actual_matrix[:, 2] = y_test
    actual_real = scaler.inverse_transform(temp_actual_matrix)[:, 2]
    
    # Calculate performance scores
    mae = mean_absolute_error(actual_real, predictions_real)
    mse = mean_squared_error(actual_real, predictions_real)
    rmse = np.sqrt(mse)
    cv_rmse = calc_cvrmse(rmse, actual_real)
    nmbe = calc_nmbe(actual_real, predictions_real)
    mda = calc_mda(actual_real, predictions_real)
    r2 = r2_score(actual_real, predictions_real)
    
    # Log everything out cleanly
    logger.info("==========================================================")
    logger.info("                   FINAL PERFORMANCE METRICS               ")
    logger.info("==========================================================")
    logger.info(f"Mean Absolute Error (MAE)        : {mae:.4f} V  [MAIN METRIC]")
    logger.info(f"Root Mean Squared Error (RMSE)   : {rmse:.4f} V")
    logger.info(f"Coefficient of Var RMSE (cvRMSE) : {cv_rmse:.2f} %")
    logger.info(f"Normalized Mean Bias Err (NMBE)  : {nmbe:.2f} %")
    logger.info(f"Mean Directional Accuracy (MDA)  : {mda:.4f}")
    logger.info(f"Coefficient of Determination(R²) : {r2:.4f}")
    logger.info("==========================================================")

    # 4. Save metrics and parameters cleanly to a text file
    try:
        with open(RESULTS_FILE, "w") as f:
            f.write("==================================================\n")
            f.write(f"  XGBOOST OPTIMIZATION & METRICS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("==================================================\n\n")
            f.write("[BEST HYPERPARAMETERS]\n")
            for key, val in best_hps.items():
                f.write(f"{key}: {val}\n")
            
            f.write("\n[VALIDATION METRICS]\n")
            f.write(f"MAE (Main Metric) : {mae:.6f} V\n")
            f.write(f"RMSE              : {rmse:.6f} V\n")
            f.write(f"cvRMSE            : {cv_rmse:.4f} %\n")
            f.write(f"NMBE              : {nmbe:.4f} %\n")
            f.write(f"MDA               : {mda:.4f}\n")
            f.write(f"R2 Score          : {r2:.6f}\n")
            f.write("==================================================\n")
        logger.info(f"Successfully saved optimized parameters and metrics to '{RESULTS_FILE}'")
    except Exception as e:
        logger.error(f"Failed to write hyperparameters text file: {e}")

    # 5. Render outputs
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    
    # For XGBoost, plot Feature Importance instead of Training Loss
    xgb.plot_importance(best_model, max_num_features=10, ax=axes[0], importance_type='weight')
    axes[0].set_title("Top 10 Flattened Feature Importances", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Actual vs Prediction Comparison
    test_dates = df.index[-len(y_test):]
    axes[1].plot(test_dates, actual_real, label='Actual Reference Potential', color='#10b981', linewidth=2)
    axes[1].plot(test_dates, predictions_real, label='XGBoost Forecast', color='#ef4444', linestyle='--', linewidth=2)
    axes[1].set_title(f"Actual vs Prediction Comparison (MAE: {mae:.3f} | cvRMSE: {cv_rmse:.1f}%)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Electrode_V")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_img = "iccp_xgboost_optimized_results.png"
    plt.savefig(output_img)
    logger.info(f"Analysis saved as: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    main()