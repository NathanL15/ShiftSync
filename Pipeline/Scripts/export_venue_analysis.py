import polars as pl
import psycopg2
from psycopg2.extras import execute_values
from model_segmentation import analyze_peak_hours

# Database connection details
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ShiftSyncDB"
DB_USER = "postgres"
DB_PASSWORD = "shiftSyncDB15"

def process_data():
    """Load and preprocess the data"""
    # Read data from CSV file
    filtered_df = pl.read_csv("Data/cleansed_data.csv").select(
        ["order_seated_at_local", "order_uuid", "venue_xref_id", "business_date", "concept"]
    )

    # Remove rows with None concept
    filtered_df = filtered_df.filter(pl.col("concept").is_not_null())

    # Convert to datetime and extract hour
    filtered_df = filtered_df.with_columns([
        pl.col("order_seated_at_local").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.lit(1).alias("order_count")
    ]).with_columns(
        pl.col("order_seated_at_local").dt.hour().alias("hour")
    )

    # Group by venue, business date, and hour, then sum the order counts
    hourly_order_counts = (
        filtered_df
        .group_by(["venue_xref_id", "business_date", "hour"])
        .agg(pl.sum("order_count").alias("order_count"))
    )

    # Calculate the average order count per hour for each venue
    average_hourly_order_counts = (
        hourly_order_counts
        .join(
            filtered_df.select(["venue_xref_id", "concept"]).unique(),
            on="venue_xref_id"
        )
        .group_by(["venue_xref_id", "concept", "hour"])
        .agg(pl.mean("order_count").alias("normalized_order_count"))
        .sort(["venue_xref_id", "hour"])
    )

    return average_hourly_order_counts

def export_to_postgres(results):
    """Export analysis results to PostgreSQL"""
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
        CREATE TABLE IF NOT EXISTS venue_peak_hours (
            id SERIAL PRIMARY KEY,
            venue_xref_id TEXT,
            concept TEXT,
            hour INTEGER,
            method TEXT,
            normalized_order_count FLOAT,
            is_peak BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS venue_overlap_analysis (
            id SERIAL PRIMARY KEY,
            venue_xref_id TEXT,
            concept TEXT,
            hour INTEGER,
            overlap_count INTEGER,
            overlap_category TEXT,
            normalized_order_count FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # For each concept in results
        for concept, concept_results in results.items():
            # Export peak hours for each method
            for method in ['kmeans', 'gmm', 'agglo']:
                if method in concept_results:
                    peak_hours = concept_results[method]
                    for row in peak_hours.iter_rows(named=True):
                        cursor.execute("""
                        INSERT INTO venue_peak_hours 
                        (concept, hour, method, normalized_order_count, is_peak)
                        VALUES (%s, %s, %s, %s, %s)
                        """, (concept, row['hour'], method, row['normalized_order_count'], True))

            # Export overlap analysis
            if 'overlap_analysis' in concept_results:
                overlap_df = concept_results['overlap_analysis']
                for row in overlap_df.iter_rows(named=True):
                    cursor.execute("""
                    INSERT INTO venue_overlap_analysis 
                    (concept, hour, overlap_count, overlap_category, normalized_order_count)
                    VALUES (%s, %s, %s, %s, %s)
                    """, (concept, row['hour'], row['overlap_count'], 
                          row['overlap_category'], row['normalized_order_count']))

        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()
        print("Data exported successfully to PostgreSQL.")
        
    except Exception as e:
        print(f"Error exporting to PostgreSQL: {e}")

def main():
    # Process the data
    print("Processing data...")
    data = process_data()
    
    # Run analysis
    print("Running analysis...")
    results = analyze_peak_hours(data, kmeans=True, gmm=True, agglo=True, 
                               overlap=True, graph=False)
    
    # Export to PostgreSQL
    print("Exporting to PostgreSQL...")
    export_to_postgres(results)
    
    print("\nAnalysis complete! Data is now available in PostgreSQL.")
    print("Tables created:")
    print("1. venue_peak_hours - Contains peak hours by venue and method")
    print("2. venue_overlap_analysis - Contains overlap analysis for each venue")

if __name__ == "__main__":
    main() 