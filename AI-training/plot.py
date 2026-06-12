import os
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURATION ---
INPUT_CSV = "device_001_data.csv"
OUTPUT_PLOT_IMAGE = "device_001_metrics_plot.png"

def plot_csv_data():
    # Check if the file actually exists first
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find '{INPUT_CSV}'. Please run your InfluxDB exporter script first.")
        return

    print(f"Reading data from {INPUT_CSV}...")
    # Load the CSV data
    df = pd.read_csv(INPUT_CSV)

    # Convert the time column to proper datetime objects so matplotlib scales the timeline correctly
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Identify the feature columns we want to plot (excluding metadata like device_id)
    features = [col for col in df.columns if col != "device_id"]
    num_features = len(features)

    if num_features == 0:
        print("Error: No valid feature columns found in the CSV to plot.")
        return

    print(f"Generating stacked plots for {num_features} features...")

    # Create a layout where each feature gets its own row/subplot stacked vertically
    # sharex=True locks all subplots to the exact same timeline scroll
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(12, 2.5 * num_features), sharex=True)

    # If there's only 1 feature, matplotlib returns a single axis object instead of a list/array.
    # We force it into a list structure so our loop works universally.
    if num_features == 1:
        axes = [axes]

    # Define a custom color palette to make the individual plots distinct
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Loop through each column and draw its timeline plot
    for i, column_name in enumerate(features):
        ax = axes[i]
        color = colors[i % len(colors)] # Cycle colors if you have more features than colors

        # Plot the data line
        ax.plot(df.index, df[column_name], color=color, linewidth=1.5, label=column_name)
        
        # Format individual subplot text and styles
        ax.set_ylabel(column_name, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)
        
        # Subtle cleanup: remove redundant top/right borders to make it modern
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Apply global labels and formatting adjustments
    plt.suptitle(f"Historical Metrics Timeline - Device 001", fontsize=14, fontweight="bold", y=0.98)
    plt.xlabel("Time", fontsize=12, labelpad=10)
    
    # Auto-rotates timestamps neatly so they never crash into each other
    fig.autofmt_xdate()
    plt.tight_layout()

    # Save to disk as a high-quality image file
    print(f"Saving plot visualization to {OUTPUT_PLOT_IMAGE}...")
    plt.savefig(OUTPUT_PLOT_IMAGE, dpi=300)
    plt.close()
    
    print("Success! Your plot image is ready.")

if __name__ == "__main__":
    plot_csv_data()