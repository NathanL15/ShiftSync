import numpy as np
import pandas as pd
import os
import polars as pl
import sqlite3
from scipy.signal import savgol_filter

# Set the directory to the folder this script is in
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(script_directory, '..'))  # Go up one directory

# Print all folders in the current directory
folders = [f for f in os.listdir() if os.path.isdir(f)]
print("Folders in the current directory:", folders)

# Load data using Polars
df = pl.read_csv('Data/bills.csv')
venues_df = pl.read_csv('Data/venues.csv')

# Merge the DataFrames on the 'venue_xref_id' column
df = df.join(venues_df, on='venue_xref_id', how='left')

# Convert order duration from seconds to minutes
df = df.with_columns((df["order_duration_seconds"] / 60).alias("order_duration_minutes"))

# Function to find the non-linear point in the quantile curve
def find_sensitive_threshold(df, order_type="dinein", column_name='order_duration_minutes', 
                             quantile_min=0.935, quantile_max=0.985, quantile_step=0.001,
                             window_length=7, polyorder=2, method='second_derivative'):
    """
    Finds threshold using more sensitive methods to detect earlier curve changes.
    """
    type_df = df.filter(pl.col("order_take_out_type_label") == order_type)
    
    # Calculate quantiles
    quantiles = np.arange(quantile_min, quantile_max, quantile_step)
    quantile_values = [df.select(pl.col(column_name).quantile(q)).item() for q in quantiles]
    
    # Calculate slopes
    slopes = np.diff(quantile_values) / np.diff(quantiles)
    
    # Apply Savitzky-Golay filter to smooth the slopes
    smooth_slopes = savgol_filter(slopes, window_length=window_length, polyorder=polyorder)
    
    # Detection methods
    if method == 'second_derivative':
        second_derivatives = np.diff(smooth_slopes)
        smooth_second_derivatives = savgol_filter(second_derivatives, window_length=min(window_length, len(second_derivatives) - 2), polyorder=min(polyorder, 1))
        normalized_second_derivatives = smooth_second_derivatives / np.max(np.abs(smooth_second_derivatives))
        start_idx = int(len(normalized_second_derivatives) * 0.1)
        threshold = 0.15  # 15% of max second derivative
        
        for i in range(start_idx, len(normalized_second_derivatives)):
            if normalized_second_derivatives[i] > threshold:
                non_linear_index = i + 1
                break
        else:
            non_linear_index = np.argmax(smooth_second_derivatives) + 1
    
    non_linear_index = min(max(0, non_linear_index), len(quantiles) - 2)
    non_linear_quantile = quantiles[non_linear_index]
    non_linear_value = quantile_values[non_linear_index]
    
    return {
        "non_linear_quantile": non_linear_quantile,
        "non_linear_value": non_linear_value
    }

# Group the data by order type and find the threshold for each group
order_types = df["order_take_out_type_label"].unique()
thresholds_per_type = {}
filtered_datasets = []

for order_type in order_types:
    type_df = df.filter(pl.col("order_take_out_type_label") == order_type)
    threshold_result = find_sensitive_threshold(type_df, order_type=order_type, column_name='order_duration_minutes', 
                                                window_length=15, polyorder=3)
    if threshold_result is not None:
        thresholds_per_type[order_type] = threshold_result
        filtered_datasets.append(type_df)
    else:
        print(f"Warning: No result returned for order type '{order_type}'. Skipping.")

# Combine all filtered datasets into one
combined_filtered_df = pl.concat(filtered_datasets)

# Save to SQLite
db_path = "Data/cleansed_data.db"
table_name = "cleansed_orders"

# Connect to SQLite database
conn = sqlite3.connect(db_path)

# Convert Polars DataFrame to Pandas (since sqlite3 works with Pandas)
combined_filtered_df_pandas = combined_filtered_df.to_pandas()

# Save to SQLite
combined_filtered_df_pandas.to_sql(table_name, conn, if_exists="replace", index=False)

# Close connection
conn.close()

print(f"Filtered dataset saved to SQLite database: {db_path}, table: {table_name}")
