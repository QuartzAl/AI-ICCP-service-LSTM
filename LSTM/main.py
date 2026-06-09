import os
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from datetime import datetime, timedelta, timezone
import joblib

# APScheduler for handling asynchronous background workers
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# InfluxDB Client supporting Flux
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv
# ==========================================
# 1. CONFIGURATION & CORE ENVIRONMENT
# ==========================================
load_dotenv()
INFLUX_URL = os.getenv("INFLUXDB_URL")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = "petra"
INFLUX_BUCKET = "ICCP"
DEVICE_ID = "001"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

# Features array mapping
FEATURES = ["bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"]
TARGET_INDEX = FEATURES.index("electrode_V")

# Specific memory cache sequence parameters (heterogeneous lookbacks)
LOOKBACK_30S = 45
LOOKBACK_2M  = 45
LOOKBACK_10M = 90

CACHE_30S = deque(maxlen=LOOKBACK_30S)
CACHE_2M  = deque(maxlen=LOOKBACK_2M)
CACHE_10M = deque(maxlen=LOOKBACK_10M)

MODELS_DIR = "./models"
tick_count = 0

# Dynamic helper to generate the Flux columns string natively
flux_columns = "[" + "\"_time\"," + ", ".join([f'"{f}"' for f in FEATURES]) + "]"

# ==========================================
# 2. MODEL AND SCALER LOADER (WITH FALLBACKS)
# ==========================================
print("📦 Loading all 3 Keras models and joblib scalers from disk...")

# Model 1: 30-second window model (45 lookback)
try:
    MODEL_30S = tf.keras.models.load_model(os.path.join(MODELS_DIR, "iccp_lstm_model_30s.keras"))
    SCALER_30S = joblib.load(os.path.join(MODELS_DIR, "iccp_scaler_30s.joblib"))
except Exception as e:
    print(f"ℹ 30s custom files not found, falling back to base names: {e}")
    MODEL_30S = tf.keras.models.load_model(os.path.join(MODELS_DIR, "iccp_lstm_model.keras"))
    SCALER_30S = joblib.load(os.path.join(MODELS_DIR, "iccp_scaler.joblib"))

# Model 2: 2-minute window model (45 lookback)
try:
    MODEL_2M = tf.keras.models.load_model(os.path.join(MODELS_DIR, "iccp_lstm_model_2m.keras"))
    SCALER_2M = joblib.load(os.path.join(MODELS_DIR, "iccp_scaler_2m.joblib"))
except Exception as e:
    print(f"ℹ 2m custom files not found, falling back to base names: {e}")
    MODEL_2M = tf.keras.models.load_model(os.path.join(MODELS_DIR, "iccp_lstm_model.keras"))
    SCALER_2M = joblib.load(os.path.join(MODELS_DIR, "iccp_scaler.joblib"))

# Model 3: 10-minute window model (90 lookback)
try:
    MODEL_10M = tf.keras.models.load_model(os.path.join(MODELS_DIR, "iccp_lstm_model_10m.keras"))
    SCALER_10M = joblib.load(os.path.join(MODELS_DIR, "iccp_scaler_10m.joblib"))
except Exception as e:
    print(f"ℹ 10m custom files not found, falling back to base names: {e}")
    MODEL_10M = tf.keras.models.load_model(os.path.join(MODELS_DIR, "iccp_lstm_model.keras"))
    SCALER_10M = joblib.load(os.path.join(MODELS_DIR, "iccp_scaler.joblib"))

print("✅ All models and scalers loaded successfully.")

# ==========================================
# 3. UTILITIES & DATABASE PIPELINE
# ==========================================

def execute_flux_query(flux_script: str) -> pd.DataFrame:
    """
    Executes a Flux query against the InfluxDB engine and outputs a Pandas DataFrame.
    """
    result = query_api.query_data_frame(flux_script)
    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        df = pd.concat(result)
    else:
        df = result
    return df


def generate_bootstrap_query(range_start: str, window_period: str) -> str:
    """
    Helper function to generate clean, consistent Flux queries with aggregateWindows.
    """
    return f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: {window_period}, fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {flux_columns})
    '''


def process_and_align_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes raw InfluxDB query responses: sets the timestamp index,
    structures exact ML features, and forward/backward fills gaps.
    """
    df['_time'] = pd.to_datetime(df['_time'])
    df.set_index('_time', inplace=True)
    df.sort_index(inplace=True)
    df = df.ffill().bfill()

    # Guarantee all features exist in the table columns
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df_aligned = df[FEATURES].ffill().bfill().fillna(0.0)
    return df_aligned


def bootstrap_all_caches():
    """
    Primes all three sequence caches instantly at service startup using historical queries.
    """
    global CACHE_30S, CACHE_2M, CACHE_10M
    print("🚀 Priming all memory caches via historical Flux queries...")

    # 1. Bootstrap 30-Second Cache (Requires past ~30 minutes for 45 steps of 30s)
    df_30s = execute_flux_query(generate_bootstrap_query("-30m", "30s"))
    if not df_30s.empty:
        df_aligned_30s = process_and_align_dataframe(df_30s).tail(LOOKBACK_30S)
        CACHE_30S.clear()
        for row in df_aligned_30s.values:
            CACHE_30S.append(row)

    # 2. Bootstrap 2-Minute Cache (Requires past ~100 minutes for 45 steps of 2m)
    df_2m = execute_flux_query(generate_bootstrap_query("-100m", "2m"))
    if not df_2m.empty:
        df_aligned_2m = process_and_align_dataframe(df_2m).tail(LOOKBACK_2M)
        CACHE_2M.clear()
        for row in df_aligned_2m.values:
            CACHE_2M.append(row)

    # 3. Bootstrap 10-Minute Cache (Requires past ~18 hours for 90 steps of 10m)
    df_10m = execute_flux_query(generate_bootstrap_query("-18h", "10m"))
    if not df_10m.empty:
        df_aligned_10m = process_and_align_dataframe(df_10m).tail(LOOKBACK_10M)
        CACHE_10M.clear()
        for row in df_aligned_10m.values:
            CACHE_10M.append(row)

    print(f"✅ Bootstrap complete. Cached items -> 30s: {len(CACHE_30S)}/{LOOKBACK_30S}, 2m: {len(CACHE_2M)}/{LOOKBACK_2M}, 10m: {len(CACHE_10M)}/{LOOKBACK_10M}")


def run_keras_inference(model, scaler, cache_deque):
    """
    Transforms cache history into standard tensors, performs Keras model
    prediction, and inverse-scales the output back to actual physical metrics.
    """
    raw_window = np.array(cache_deque)
    scaled_window = scaler.transform(raw_window)

    input_tensor = np.expand_dims(scaled_window, axis=0)
    scaled_prediction = model.predict(input_tensor, verbose=0)[0][0]

    dummy_row = np.zeros((1, len(FEATURES)))
    dummy_row[0, TARGET_INDEX] = scaled_prediction

    inverse_row = scaler.inverse_transform(dummy_row)
    return inverse_row[0, TARGET_INDEX]

# ==========================================
# 4. UNIFIED TIME SERIES PIPELINE STEP
# ==========================================

def execute_pipeline_tick():
    """
    Core orchestrator task running exactly every 30 seconds.
    Directs in-memory downsampling and triggers ML inferences dynamically.
    """
    global CACHE_30S, CACHE_2M, CACHE_10M, tick_count
    tick_count += 1

    print(f"\n⏱️ Tick {tick_count} executed at: {datetime.now().strftime('%H:%M:%S')}")

    # 1. Pull ONLY the single newest 30-second window
    query_30s = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -5m)
      |> filter(fn: (r) => r["device_id"] == "{DEVICE_ID}")
      |> aggregateWindow(every: 30s, fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: {flux_columns})
      |> tail(n: 1)
    '''

    try:
        df = execute_flux_query(query_30s)
        if df.empty:
            print("⚠ Flux returned no new point for this tick cycle.")
            return

        # Ensure structural shape integrity
        for col in FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        df_aligned = df[FEATURES]

        # Fallback to last known cache values if NaN values exist
        if df_aligned.isnull().values.any() and len(CACHE_30S) > 0:
            last_historical_row = CACHE_30S[-1]
            latest_features = np.where(df_aligned.isna().values[0], last_historical_row, df_aligned.values[0])
        else:
            latest_features = df_aligned.values[0]

        # Append the new 30s feature values to cache
        CACHE_30S.append(latest_features)

        # --- [MODEL 1: 30s Aggregation -> 7m Forecast] ---
        if len(CACHE_30S) >= LOOKBACK_30S:
            pred_30s = run_keras_inference(MODEL_30S, SCALER_30S, CACHE_30S)
            future_time_30s = datetime.now(timezone.utc) + timedelta(minutes=7)

            p_30s = Point("sensor_measurement") \
                .tag("device_id", DEVICE_ID) \
                .tag("type", "forecast") \
                .tag("horizon", "7m") \
                .tag("model", "keras_30s_service") \
                .field("electrode_V", float(pred_30s)) \
                .time(future_time_30s)

            write_api.write(bucket=INFLUX_BUCKET, record=p_30s)
            print(f"🔮 [7m Horizon] Predicted electrode_V: {pred_30s:.4f}V | Target Timestamp: {future_time_30s.strftime('%H:%M:%S')}")
        else:
            print(f"ℹ CACHE_30S filling: {len(CACHE_30S)}/{LOOKBACK_30S}")

        # --- [MODEL 2: 2m Aggregation -> 30m Forecast] ---
        # Executes every 4 ticks of the 30s pipeline (every 2 minutes)
        if tick_count % 4 == 0:
            # Downsample the last 4 ticks of the 30-second series in Python memory
            recent_30s_points = list(CACHE_30S)[-4:]
            two_min_mean = np.mean(recent_30s_points, axis=0)
            CACHE_2M.append(two_min_mean)

            if len(CACHE_2M) >= LOOKBACK_2M:
                pred_2m = run_keras_inference(MODEL_2M, SCALER_2M, CACHE_2M)
                future_time_2m = datetime.now(timezone.utc) + timedelta(minutes=30)

                p_2m = Point("sensor_measurement") \
                    .tag("device_id", DEVICE_ID) \
                    .tag("type", "forecast") \
                    .tag("horizon", "30m") \
                    .tag("model", "keras_2m_service") \
                    .field("electrode_V", float(pred_2m)) \
                    .time(future_time_2m)

                write_api.write(bucket=INFLUX_BUCKET, record=p_2m)
                print(f"🔮 [30m Horizon] Predicted electrode_V: {pred_2m:.4f}V | Target Timestamp: {future_time_2m.strftime('%H:%M:%S')}")
            else:
                print(f"ℹ CACHE_2M filling: {len(CACHE_2M)}/{LOOKBACK_2M}")

        # --- [MODEL 3: 10m Aggregation -> 12h Forecast] ---
        # Executes every 20 ticks of the 30s pipeline (every 10 minutes / 5 elements of CACHE_2M)
        if tick_count % 20 == 0:
            # Downsample the last 5 elements of the 2-minute series in Python memory (5 * 2m = 10m)
            recent_2m_points = list(CACHE_2M)[-5:]
            ten_min_mean = np.mean(recent_2m_points, axis=0)
            CACHE_10M.append(ten_min_mean)

            if len(CACHE_10M) >= LOOKBACK_10M:
                pred_10m = run_keras_inference(MODEL_10M, SCALER_10M, CACHE_10M)
                future_time_10m = datetime.now(timezone.utc) + timedelta(hours=12)

                p_10m = Point("sensor_measurement") \
                    .tag("device_id", DEVICE_ID) \
                    .tag("type", "forecast") \
                    .tag("horizon", "12h") \
                    .tag("model", "keras_10m_service") \
                    .field("electrode_V", float(pred_10m)) \
                    .time(future_time_10m)

                write_api.write(bucket=INFLUX_BUCKET, record=p_10m)
                print(f"🔮 [12h Horizon] Predicted electrode_V: {pred_10m:.4f}V | Target Timestamp: {future_time_10m.strftime('%H:%M:%S')}")
            else:
                print(f"ℹ CACHE_10M filling: {len(CACHE_10M)}/{LOOKBACK_10M}")

            # Keep ticker count bound to prevent integer overflow over weeks of runtime (divisible by 4 and 20)
            if tick_count >= 2400:
                tick_count = 0

    except Exception as e:
        print(f"❌ Execution Pipeline Failure on Tick: {e}")

# ==========================================
# 5. SERVICE RUNNER INTERFACE
# ==========================================
async def main():
    # Warm up all sequence caches from database history on launch
    bootstrap_all_caches()

    # Initialize task scheduler with an asynchronous execution loop
    scheduler = AsyncIOScheduler()
    scheduler.add_job(execute_pipeline_tick, 'interval', seconds=30)
    scheduler.start()

    print("🟢 Unified ML Time Series Service Active. Running orchestration loop every 30 seconds...")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("🛑 Initiating graceful shutdown of ML background service.")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
