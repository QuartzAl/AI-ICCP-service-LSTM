import os
import asyncio
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timezone
from collections import deque
from dotenv import load_dotenv
import sys

# Darts & PyTorch
import torch
from darts import TimeSeries
from darts.models import TFTModel
import torchmetrics

# APScheduler for handling asynchronous background workers
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# InfluxDB Client supporting Flux
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Optimize precision for PyTorch
torch.set_float32_matmul_precision('medium')
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & CORE ENVIRONMENT
# ==========================================
load_dotenv()
INFLUX_URL = os.getenv("INFLUXDB_URL", "http://edx.petra.ac.id:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q==")
INFLUX_ORG = "petra"
INFLUX_BUCKET = "ICCP"
DEVICE_ID = "001"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=30000)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

# Feature and Time configuration
FEATURES = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]
TARGET_COL = "electrode_V"
PAST_COV_COLS = ["bus_voltage_V", "current_mA", "soil_humidity_V"]
FUTURE_COV_COLS = ["target_current_mA"]

AGG_WINDOW = "5m"
PANDAS_FREQ = "5min"
PREDICTION_GAP = 576  # Predict 48 hours into the future (576 steps * 5 mins)

MODELS_DIR = "./models/2day-gap/"
tick_count = 0

flux_columns = "[" + "\"_time\"," + ", ".join([f'"{f}"' for f in FEATURES]) + "]"

# ==========================================
# 2. LOAD MODEL AND SCALERS
# ==========================================
print("📦 Loading TFT Model and Scalers from disk...")
# --- 1. Version Mismatch Compatibility Patch ---
# Dynamically register EarlyStoppingReason on the local PyTorch Lightning modules 
# so unpickling works flawlessly on older versions of lightning.
try:
    import pytorch_lightning.callbacks.early_stopping
    if not hasattr(pytorch_lightning.callbacks.early_stopping, 'EarlyStoppingReason'):
        from enum import Enum
        class EarlyStoppingReason(Enum):
            NOT_STOPPED = 0
            STOPPING_THRESHOLD = 1
            DIVERGENCE_THRESHOLD = 2
            PATIENCE_EXHAUSTED = 3
            NON_FINITE_METRIC = 4
        pytorch_lightning.callbacks.early_stopping.EarlyStoppingReason = EarlyStoppingReason
        
        # Patch lightning.pytorch namespace too if it exists
        try:
            import lightning.pytorch.callbacks.early_stopping
            if not hasattr(lightning.pytorch.callbacks.early_stopping, 'EarlyStoppingReason'):
                lightning.pytorch.callbacks.early_stopping.EarlyStoppingReason = EarlyStoppingReason
        except ImportError:
            pass
except Exception as patch_err:
    print(f"⚠️ Warning during EarlyStoppingReason patch: {patch_err}")

try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        import torch.optim
        torch.serialization.add_safe_globals([torch.optim.Adam, torch.nn.modules.loss.HuberLoss, torchmetrics.collections.MetricCollection])
    # TFTModel.load automatically finds the adjacent .ckpt weights
    MODEL = TFTModel.load(os.path.join(MODELS_DIR, "best_tft_model.pt"))
    
    with open(os.path.join(MODELS_DIR, "scalers.pkl"), "rb") as f:
        SCALERS = pickle.load(f)
        
    TARGET_SCALER = SCALERS["target_scaler"]
    PAST_COV_SCALER = SCALERS["past_cov_scaler"]
    FUTURE_COV_SCALER = SCALERS["future_cov_scaler"]
    
    # Dynamically read the optimal chunk lengths from the trained model
    IN_CHUNK = MODEL.input_chunk_length
    OUT_CHUNK = MODEL.output_chunk_length
    
    print(f"✅ TFT Model loaded successfully! (Input Window: {IN_CHUNK}, Optimal Output Chunk: {OUT_CHUNK})")
except Exception as e:
    print(f"❌ Failed to load TFT Model or Scalers: {e}")
    sys.exit(1)

# ==========================================
# 3. MEMORY CACHING SETUP (DEQUE)
# ==========================================
# The cache must hold enough data to satisfy the input chunk, the prediction gap shift, 
# and a few extra steps for differencing and safety.
CACHE_MAXLEN = IN_CHUNK + PREDICTION_GAP + 10
CACHE = deque(maxlen=CACHE_MAXLEN)

def execute_flux_query(flux_script: str) -> pd.DataFrame:
    result = query_api.query_data_frame(flux_script)
    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        df = pd.concat(result)
    else:
        df = result
    return df

def bootstrap_cache():
    """
    Primes the memory cache instantly at service startup using a historical query.
    """
    global CACHE
    print(f"🚀 Priming memory cache (Need {CACHE_MAXLEN} rows)...")
    
    # Calculate how many minutes of history we need to fill the cache
    minutes_needed = CACHE_MAXLEN * int(AGG_WINDOW.replace('m', ''))
    
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{minutes_needed}m)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}" and not exists r.type)
      |> aggregateWindow(every: {AGG_WINDOW}, fn: mean, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {flux_columns})
    '''
    
    df = execute_flux_query(query)
    if not df.empty:
        df['_time'] = pd.to_datetime(df['_time']).dt.tz_localize(None)
        df.set_index('_time', inplace=True)
        df.sort_index(inplace=True)
        df = df.asfreq(PANDAS_FREQ).ffill().bfill()
        
        # Guarantee all features exist
        for col in FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        df_aligned = df[FEATURES].ffill().bfill().fillna(0.0)
        
        # Populate the deque with (timestamp, dict_row) tuples
        CACHE.clear()
        for index, row in df_aligned.iterrows():
            CACHE.append((index, row.to_dict()))
            
    print(f"✅ Bootstrap complete. Cached items -> {len(CACHE)}/{CACHE_MAXLEN}")

# ==========================================
# 4. UNIFIED TIME SERIES PIPELINE STEP
# ==========================================

def execute_pipeline_tick():
    """
    Core orchestrator task running every 5 minutes.
    Updates memory cache, aligns to the TFT gap logic, predicts the optimal chunk, and pushes to InfluxDB.
    """
    global tick_count, CACHE
    tick_count += 1
    print(f"\n⏱️ Tick {tick_count} executed at: {datetime.now().strftime('%H:%M:%S')}")

    # Fetch ONLY the single newest 5-minute window
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -15m)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}" and not exists r.type)
      |> aggregateWindow(every: {AGG_WINDOW}, fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {flux_columns})
      |> tail(n: 1)
    '''

    try:
        df_new = execute_flux_query(query)
        if df_new.empty:
            print("⚠ Flux returned no data for this tick cycle.")
            return

        # Structure the new row
        df_new['_time'] = pd.to_datetime(df_new['_time']).dt.tz_localize(None)
        df_new.set_index('_time', inplace=True)
        for col in FEATURES:
            if col not in df_new.columns:
                df_new[col] = np.nan
                
        # Cast explicitly to float to prevent InfluxDB 'None' objects
        df_new = df_new[FEATURES].astype(float).ffill().bfill().fillna(0.0)

        # Append new tick to cache
        CACHE.append((df_new.index[0], df_new.iloc[0].to_dict()))

        # Guard: Ensure we have enough data before running inference
        if len(CACHE) < IN_CHUNK + PREDICTION_GAP + 2:
            print(f"ℹ Cache still filling... {len(CACHE)}/{CACHE_MAXLEN} rows available. Waiting to predict.")
            return

        # 1. Rebuild DataFrame strictly from Memory Cache (Explicitly specify dtype=float)
        df = pd.DataFrame([x[1] for x in CACHE], index=[x[0] for x in CACHE], dtype=float)
        df = df.asfreq(PANDAS_FREQ).ffill().bfill()
        current_time = df.index[-1]
        
        # Keep an unshifted copy to lookup the true present Absolute Voltage
        df_original = df.copy()

        # 2. Extend future covariates into the 'void' to cover Gap + Output Chunk
        total_future_steps = PREDICTION_GAP + OUT_CHUNK
        future_dates = pd.date_range(start=current_time + pd.Timedelta(PANDAS_FREQ), periods=total_future_steps, freq=PANDAS_FREQ)
        
        # Set dtype=float when creating the empty dataframe to prevent None objects
        df_extended = pd.DataFrame(index=future_dates, columns=FEATURES, dtype=float)
        df_combined = pd.concat([df, df_extended])
        
        # Assume control current remains identical to its last known state into the future
        df_combined['target_current_mA'] = df_combined['target_current_mA'].ffill()

        # 3. Apply the Model's Target Shift
        df_shifted = df_combined.copy()
        df_shifted[TARGET_COL] = df_shifted[TARGET_COL].shift(-PREDICTION_GAP)
        for col in FUTURE_COV_COLS:
            df_shifted[col] = df_shifted[col].shift(-PREDICTION_GAP)

        # Ensure everything is absolutely a float right before the diff
        df_shifted = df_shifted.astype(float)

        # 4. First-Order Differencing
        df_diff = df_shifted.diff()
        # 5. Extract Valid Data into TimeSeries Objects (DropNA automatically handles the slice)
        target_ts = TimeSeries.from_dataframe(df_diff[[TARGET_COL]].dropna())
        past_cov_ts = TimeSeries.from_dataframe(df_diff[PAST_COV_COLS].dropna())
        future_cov_ts = TimeSeries.from_dataframe(df_diff[FUTURE_COV_COLS].dropna())

        # 6. Scale Inputs
        target_ts_scaled = TARGET_SCALER.transform(target_ts)
        past_cov_ts_scaled = PAST_COV_SCALER.transform(past_cov_ts)
        future_cov_ts_scaled = FUTURE_COV_SCALER.transform(future_cov_ts)

        # 7. Model Inference (Predict ONLY the optimal output chunk length)
        pred_scaled = MODEL.predict(
            n=OUT_CHUNK,
            series=target_ts_scaled,
            past_covariates=past_cov_ts_scaled,
            future_covariates=future_cov_ts_scaled,
            verbose=False
        )

        # 8. Unscale and Reconstruct Absolute Voltage
        pred_deltas = TARGET_SCALER.inverse_transform(pred_scaled).values().flatten()
        
        # Anchor the cumulative sum to the actual present voltage
        base_voltage = df_original.loc[current_time, TARGET_COL]
        reconstructed_abs_volts = base_voltage + np.cumsum(pred_deltas)
        
        # 9. Write entire Output Chunk trajectory to InfluxDB
        for i, val in enumerate(reconstructed_abs_volts):
            # Target timeline begins at the Gap offset
            target_time = current_time + (pd.Timedelta(PANDAS_FREQ) * (PREDICTION_GAP + i))
            
            point = Point("sensor_measurement") \
                .tag("device_id", DEVICE_ID) \
                .tag("type", "forecast") \
                .tag("horizon", f"{(PREDICTION_GAP + i) * 5}m") \
                .tag("model", "tft_darts_service") \
                .field("electrode_V", float(val)) \
                .time(target_time.tz_localize('UTC')) 

            write_api.write(bucket=INFLUX_BUCKET, record=point)
            
        print(f"🔮 [Predicted {OUT_CHUNK} steps at Gap {PREDICTION_GAP}] Latest electrode_V: {reconstructed_abs_volts[-1]:.4f}V | Furthest Timestamp: {target_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    except Exception as e:
        print(f"❌ Execution Pipeline Failure on Tick: {e}")
        traceback.print_exc()

# ==========================================
# 5. SERVICE RUNNER INTERFACE
# ==========================================
async def main():
    # Warm up sequence cache from database history on launch
    bootstrap_cache()
    
    # Initialize task scheduler
    scheduler = AsyncIOScheduler()
    
    # Run exactly every 5 minutes
    scheduler.add_job(execute_pipeline_tick, 'interval', minutes=5)
    scheduler.start()

    print(f"🟢 Stateful TFT ML Time Series Service Active. (Agg Window: {AGG_WINDOW}, Gap: {PREDICTION_GAP})")
    print("Running orchestration loop every 5 minutes...")
    
    # Fire off an immediate tick on launch
    execute_pipeline_tick()

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Initiating graceful shutdown of ML background service.")
    finally:
        client.close()

if __name__ == "__main__":
    import traceback
    asyncio.run(main())

