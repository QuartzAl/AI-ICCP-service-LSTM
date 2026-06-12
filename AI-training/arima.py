import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from influxdb_client import InfluxDBClient
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- CONFIGURATION ---
URL = "http://edx.petra.ac.id:8086" 
TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q==" 
ORG = "petra"
BUCKET = "ICCP"
OUTPUT_IMAGE = "ARIMA.png"

# Target feature to forecast (ARIMA works on a single target column at a time)
TARGET_FEATURE = "target_current_mA"

# --- FLUX QUERY ---
# Updated range to -20d and aggregateWindow to 15m
flux_query = f"""
from(bucket: "{BUCKET}")
    |> range(start: -20d)  
    |> filter(fn: (r) => r["device_id"] == "001")
    |> filter(fn: (r) => r["_field"] == "target_current_mA" or r["_field"] == "electrode_V")
    |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
    |> keep(columns: ["_time", "_field", "_value"])
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
"""

def get_influx_data():
    print("Connecting to InfluxDB...")
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query_api = client.query_api()

    try:
        print("Fetching data...")
        df = query_api.query_data_frame(query=flux_query)

        if df.empty:
            print("No data found at all. Check your bucket/device_id connection.")
            return None

        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)

        # Establish strict Time Index
        df["_time"] = pd.to_datetime(df["_time"])
        df.set_index("_time", inplace=True)

        # Dynamic fallback columns if completely missing
        for expected_col in ["target_current_mA", "electrode_V"]:
            if expected_col not in df.columns:
                df[expected_col] = 0.0

        # Convert data columns to numeric float types
        df["target_current_mA"] = pd.to_numeric(df["target_current_mA"], errors="coerce")
        df["electrode_V"] = pd.to_numeric(df["electrode_V"], errors="coerce")

        # Isolate ONLY numeric columns before calling resample
        numeric_df = df[["target_current_mA", "electrode_V"]]
        
        # Updated resampling string grid to 15min to match InfluxDB
        final_df = numeric_df.resample("15min").mean().ffill().bfill()

        return final_df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    finally:
        client.close()

def train_sarimax_model(df):
    print("\nTraining SARIMAX model...")
    
    output_y = df["electrode_V"]            
    input_x = df[["target_current_mA"]]     

    # Standard order configurations
    order = (2, 1, 1)
    
    # Optional Seasonal parameter tune:
    # If you want to track a 24-hour cycle with 15-minute data, s would equal 96.
    # Leaving it at (0,0,0,0) for fast background calculation unless explicitly required.
    seasonal_order = (0, 0, 0, 0) 

    model = SARIMAX(output_y, exog=input_x, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    print(model_fit.summary())

    # We will forecast 8 steps forward (8 intervals * 15 mins = next 2 hours)
    forecast_steps = 8
    last_known_input = input_x.iloc[-1].values[0]
    
    future_exog = pd.DataFrame(
        {"target_current_mA": [last_known_input] * forecast_steps},
        index=pd.date_range(start=df.index[-1] + pd.Timedelta(minutes=15), periods=forecast_steps, freq="15min")
    )

    forecast = model_fit.forecast(steps=forecast_steps, exog=future_exog)

    # --- PLOTTING ---
    print(f"\nGenerating and saving plot to {OUTPUT_IMAGE}...")
    plt.figure(figsize=(12, 6))
    
    # Plotting the last 48 points (12 hours of actual history) for context
    historical_subset = output_y.tail(48)
    plt.plot(historical_subset.index, historical_subset.values, label='Actual electrode_V (Last 12 Hours)', color='#2ca02c', linewidth=2)
    plt.plot(forecast.index, forecast.values, label='SARIMAX Forecasted electrode_V (Next 2 Hours)', color='#d62728', linestyle='--', linewidth=2, marker='o')
    
    plt.title("SARIMAX Model: 15-Minute Grid Forecast for electrode_V", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Electrode Voltage (V)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.close()
    print("Plot saved successfully!")

if __name__ == "__main__":
    data = get_influx_data()
    if data is not None:
        train_sarimax_model(data)