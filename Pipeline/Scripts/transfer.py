import psycopg2
import pandas as pd
import polars as pl
from model_segmentation import analyze_peak_hours

# Database connection details
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ShiftSyncDB"
DB_USER = "postgres"
DB_PASSWORD = "shiftSyncDB15"

# Load your data
data_path = "path_to_your_data.csv"
data = pl.read_csv(data_path)

# Run the analysis
results = analyze_peak_hours(data)

# Prepare the data for export
peak_hours_df = results['peak_hours'].to_pandas()

# Function to insert data into PostgreSQL
def export_to_postgres(dataframe, table_name):
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            concept TEXT,
            method TEXT,
            hour INTEGER,
            normalized_order_count FLOAT,
            is_peak BOOLEAN
        );
        """)

        # Insert data into the table
        for _, row in dataframe.iterrows():
            cursor.execute(f"""
            INSERT INTO {table_name} (concept, method, hour, normalized_order_count, is_peak)
            VALUES (%s, %s, %s, %s, %s);
            """, (row['concept'], row['method'], row['hour'], row['normalized_order_count'], row['is_peak']))

        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()
        print("Data exported successfully to PostgreSQL.")
    except Exception as e:
        print(f"Error: {e}")

# Export the data
export_to_postgres(peak_hours_df, "graphical_analysis")