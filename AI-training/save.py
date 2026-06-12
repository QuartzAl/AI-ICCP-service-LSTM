import pandas as pd
from influxdb_client import InfluxDBClient

# --- CONFIGURATION ---
URL = "http://edx.petra.ac.id:8086" 
TOKEN = "1HMtNH3Gl2IgYq3sWnj3JaD47xemc1HSJKCjhkcSkum3lA3ueNN9KLCEDTjl41DNsfLCBBuffY0l5OHixeig6Q==" 
ORG = "petra"
BUCKET = "ICCP"  
OUTPUT_CSV = "device_001_data.csv"

# The features you want to extract
FEATURES = [
    "bus_voltage_V",
    "current_mA",
    "electrode_V",
    "soil_humidity_V",
    "target_current_mA",
]

# --- FIX: Format the Python list into a Flux literal set {"a", "b", "c"} ---
flux_set = "{" + ", ".join([f'"{f}"' for f in FEATURES]) + "}"

# --- FLUX QUERY ---
flux_query = f"""
from(bucket: "{BUCKET}")
    |> range(start: -30d)
    |> filter(fn: (r) => r["device_id"] == "001")
    |> filter(fn: (r) => r["_field"] == "bus_voltage_V" or 
                         r["_field"] == "current_mA" or 
                         r["_field"] == "electrode_V" or 
                         r["_field"] == "soil_humidity_V" or 
                         r["_field"] == "target_current_mA")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "device_id", "bus_voltage_V", "current_mA", "electrode_V", "soil_humidity_V", "target_current_mA"])
"""

def export_influx_to_csv():
    print("Connecting to InfluxDB...")
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query_api = client.query_api()

    try:
        print("Executing query and converting to DataFrame...")
        df = query_api.query_data_frame(query=flux_query)

        if df.empty:
            print("No data found for the specified criteria. Check your bucket/measurement names.")
            return

        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)

        df.rename(columns={"_time": "time"}, inplace=True)

        # Filter out to keep only the requested columns that exist in the result
        existing_columns = [col for col in ["time", "device_id"] + FEATURES if col in df.columns]
        df = df[existing_columns]

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Success! Data successfully saved to {OUTPUT_CSV}")
        print(f"Total rows exported: {len(df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        client.close()

if __name__ == "__main__":
    export_influx_to_csv()