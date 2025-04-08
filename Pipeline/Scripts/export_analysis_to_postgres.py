import os
import sys
import pandas as pd
import polars as pl
import psycopg2
from psycopg2.extras import execute_values
import json
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import from the Scripts directory
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)

# Import the model segmentation function
from Scripts.model_segmentation import analyze_peak_hours

# Database connection details - replace with your actual PostgreSQL credentials
DB_HOST = "your_postgres_host"
DB_PORT = "your_postgres_port"
DB_NAME = "your_database_name"
DB_USER = "your_username"
DB_PASSWORD = "your_password"

def load_data():
    """
    Load and preprocess the data for analysis.
    """
    # Load the cleansed data
    data_path = os.path.join(parent_directory, "Data", "cleansed_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: File '{data_path}' not found.")
        sys.exit(1)
    
    # Load the data
    df = pl.read_csv(data_path)
    
    # Ensure we have the required columns
    required_columns = ['concept', 'hour', 'normalized_order_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print("Please ensure the data has been preprocessed correctly.")
        sys.exit(1)
    
    return df

def run_analysis(data):
    """
    Run the peak hours analysis on the data.
    """
    print("Running peak hours analysis...")
    results = analyze_peak_hours(data, kmeans=True, gmm=True, agglo=True, overlap=True, graph=False)
    return results

def prepare_data_for_postgres(results):
    """
    Convert the analysis results into a format suitable for PostgreSQL.
    """
    # Create DataFrames for each type of analysis result
    
    # 1. Peak hours by method for each concept
    peak_hours_data = []
    
    for concept, concept_results in results.items():
        # KMeans results
        if 'kmeans' in concept_results:
            kmeans_df = concept_results['kmeans']
            for row in kmeans_df.iter_rows(named=True):
                peak_hours_data.append({
                    'concept': concept,
                    'method': 'kmeans',
                    'hour': row['hour'],
                    'normalized_order_count': row['normalized_order_count'],
                    'is_peak': True
                })
        
        # GMM results
        if 'gmm' in concept_results:
            gmm_df = concept_results['gmm']
            for row in gmm_df.iter_rows(named=True):
                peak_hours_data.append({
                    'concept': concept,
                    'method': 'gmm',
                    'hour': row['hour'],
                    'normalized_order_count': row['normalized_order_count'],
                    'is_peak': True
                })
        
        # Agglomerative results
        if 'agglo' in concept_results:
            agglo_df = concept_results['agglo']
            for row in agglo_df.iter_rows(named=True):
                peak_hours_data.append({
                    'concept': concept,
                    'method': 'agglo',
                    'hour': row['hour'],
                    'normalized_order_count': row['normalized_order_count'],
                    'is_peak': True
                })
    
    peak_hours_df = pd.DataFrame(peak_hours_data)
    
    # 2. Overlap analysis for each concept
    overlap_data = []
    
    for concept, concept_results in results.items():
        if 'overlap_analysis' in concept_results:
            overlap_df = concept_results['overlap_analysis']
            for row in overlap_df.iter_rows(named=True):
                overlap_data.append({
                    'concept': concept,
                    'hour': row['hour'],
                    'overlap_count': row['overlap_count'],
                    'overlap_category': row['overlap_category'],
                    'normalized_order_count': row['normalized_order_count']
                })
    
    overlap_df = pd.DataFrame(overlap_data)
    
    # 3. Create a summary table with the most reliable peak hours (high agreement)
    summary_data = []
    
    for concept, concept_results in results.items():
        if 'overlap_analysis' in concept_results:
            overlap_df = concept_results['overlap_analysis']
            high_agreement_hours = overlap_df.filter(pl.col('overlap_count') >= 2)
            
            for row in high_agreement_hours.iter_rows(named=True):
                summary_data.append({
                    'concept': concept,
                    'hour': row['hour'],
                    'agreement_level': row['overlap_category'],
                    'agreement_count': row['overlap_count'],
                    'normalized_order_count': row['normalized_order_count']
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    return {
        'peak_hours': peak_hours_df,
        'overlap_analysis': overlap_df,
        'summary': summary_df
    }

def export_to_postgres(data_dict):
    """
    Export the analysis results to PostgreSQL.
    """
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
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS peak_hours_analysis (
            id SERIAL PRIMARY KEY,
            concept TEXT,
            method TEXT,
            hour INTEGER,
            normalized_order_count FLOAT,
            is_peak BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS overlap_analysis (
            id SERIAL PRIMARY KEY,
            concept TEXT,
            hour INTEGER,
            overlap_count INTEGER,
            overlap_category TEXT,
            normalized_order_count FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS peak_hours_summary (
            id SERIAL PRIMARY KEY,
            concept TEXT,
            hour INTEGER,
            agreement_level TEXT,
            agreement_count INTEGER,
            normalized_order_count FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Insert data into the tables
        # 1. Peak hours analysis
        if not data_dict['peak_hours'].empty:
            peak_hours_data = [tuple(x) for x in data_dict['peak_hours'].to_numpy()]
            execute_values(
                cursor,
                """
                INSERT INTO peak_hours_analysis 
                (concept, method, hour, normalized_order_count, is_peak)
                VALUES %s
                """,
                peak_hours_data
            )
        
        # 2. Overlap analysis
        if not data_dict['overlap_analysis'].empty:
            overlap_data = [tuple(x) for x in data_dict['overlap_analysis'].to_numpy()]
            execute_values(
                cursor,
                """
                INSERT INTO overlap_analysis 
                (concept, hour, overlap_count, overlap_category, normalized_order_count)
                VALUES %s
                """,
                overlap_data
            )
        
        # 3. Summary
        if not data_dict['summary'].empty:
            summary_data = [tuple(x) for x in data_dict['summary'].to_numpy()]
            execute_values(
                cursor,
                """
                INSERT INTO peak_hours_summary 
                (concept, hour, agreement_level, agreement_count, normalized_order_count)
                VALUES %s
                """,
                summary_data
            )
        
        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()
        print("Data exported successfully to PostgreSQL.")
        
    except Exception as e:
        print(f"Error exporting to PostgreSQL: {e}")

def main():
    """
    Main function to run the analysis and export to PostgreSQL.
    """
    # Load data
    data = load_data()
    
    # Run analysis
    results = run_analysis(data)
    
    # Prepare data for PostgreSQL
    data_for_postgres = prepare_data_for_postgres(results)
    
    # Export to PostgreSQL
    export_to_postgres(data_for_postgres)
    
    print("\nExport completed successfully!")
    print("\nTo connect PowerBI to this PostgreSQL database:")
    print("1. Open PowerBI Desktop")
    print("2. Click 'Get Data' > 'PostgreSQL database'")
    print("3. Enter the server details: " + DB_HOST)
    print("4. Enter the database name: " + DB_NAME)
    print("5. Select 'Import' or 'DirectQuery' mode")
    print("6. Choose the tables: peak_hours_analysis, overlap_analysis, and peak_hours_summary")
    print("7. Create visualizations based on the data")

if __name__ == "__main__":
    main() 