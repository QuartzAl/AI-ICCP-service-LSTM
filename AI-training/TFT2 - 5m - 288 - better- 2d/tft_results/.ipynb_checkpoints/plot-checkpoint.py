import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def plot_and_evaluate(csv_filepath, n_shift=0):
    # 1. Load the data
    df = pd.read_csv(csv_filepath)
    
    # 2. Shift the prediction column by 'n'
    df['Shifted_TFT_Prediction'] = df['TFT_Prediction_V_Abs'].shift(n_shift)
    
    # ==========================================
    # 3. CALCULATE METRICS
    # ==========================================
    # Drop rows where the shift caused NaN values so metrics don't break
    df_metrics = df[['Actual_Electrode_V_Abs', 'Shifted_TFT_Prediction']].dropna()
    
    y_true = df_metrics['Actual_Electrode_V_Abs']
    y_pred = df_metrics['Shifted_TFT_Prediction']
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE) and cvRMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_actual = np.mean(y_true)
    
    # cvRMSE is usually expressed as a percentage: (RMSE / Mean of Actuals) * 100
    if mean_actual != 0:
        cvrmse = (rmse / mean_actual) * 100
    else:
        cvrmse = float('inf') # Handle division by zero just in case

    print(f"--- Metrics for n_shift = {n_shift} ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"MAE:            {mae:.4f}")
    print(f"cvRMSE:         {cvrmse:.4f}%")
    print("-" * 33)
    
    # ==========================================
    # 4. PLOT THE DATA
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['Real_Future_Timestamp'], df['Actual_Electrode_V_Abs'], 
             label='Actual Electrode V Abs', color='blue', linewidth=2)
    
    plt.plot(df['Real_Future_Timestamp'], df['Shifted_TFT_Prediction'], 
             label=f'TFT Prediction V Abs (Shifted by n={n_shift})', 
             color='orange', linestyle='--', linewidth=2)
    
    # Format the plot
    plt.title(f'Actual vs Shifted Predicted Electrode Voltage (Shift={n_shift})\nMAE: {mae:.3f} | cvRMSE: {cvrmse:.2f}%')
    plt.xlabel('Real Future Timestamp')
    plt.ylabel('Voltage (Abs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Fix overlapping x-axis labels
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('shifted_plot_with_metrics.png')
    plt.show()

# --- Example Usage ---
# Replace 'data.csv' with your actual file name and set your desired shift 'n'
# plot_shifted_predictions('data.csv', n_shift=5)
if __name__ == "__main__":
    plot_and_evaluate("timeseries_predictions.csv", -576)