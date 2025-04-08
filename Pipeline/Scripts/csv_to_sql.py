import os
import sqlite3
import pandas as pd

def csv_to_sqlite(csv_path):
    """
    Converts a CSV file to a SQLite database in the same folder.
    
    - The database file will have the same name as the CSV file.
    - The table inside the database will be named after the CSV filename.
    
    Parameters:
        csv_path (str): The full path to the CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    # Extract filename and directory
    folder, filename = os.path.split(csv_path)
    name, _ = os.path.splitext(filename)

    # Define SQLite database path (same folder, same name)
    sqlite_path = os.path.join(folder, f"{name}.db")
    
    # Read CSV into Pandas DataFrame
    df = pd.read_csv(csv_path)

    # Connect to SQLite and save the DataFrame
    conn = sqlite3.connect(sqlite_path)
    df.to_sql(name, conn, if_exists="replace", index=False)
    conn.close()  

    print(f"✅ Converted '{csv_path}' to SQLite: {sqlite_path}, Table: '{name}'")

# Example usage
csv_to_sqlite("Data/bills.csv")  # Change this to any CSV path
