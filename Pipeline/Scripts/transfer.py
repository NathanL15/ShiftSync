import psycopg2
import pandas as pd

# Database connection details
DB_HOST = "your_postgres_host"
DB_PORT = "your_postgres_port"
DB_NAME = "your_database_name"
DB_USER = "your_username"
DB_PASSWORD = "your_password"

# Example data to export (replace with your graphical analysis data)
data = {
    "category": ["A", "B", "C"],
    "value": [100, 200, 300]
}
df = pd.DataFrame(data)

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
            category TEXT,
            value INT
        );
        """)

        # Insert data into the table
        for _, row in dataframe.iterrows():
            cursor.execute(f"""
            INSERT INTO {table_name} (category, value)
            VALUES (%s, %s);
            """, (row['category'], row['value']))

        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()
        print("Data exported successfully to PostgreSQL.")
    except Exception as e:
        print(f"Error: {e}")

# Export the data
export_to_postgres(df, "graphical_analysis")