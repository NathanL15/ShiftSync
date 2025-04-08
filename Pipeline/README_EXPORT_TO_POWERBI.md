# Exporting Analysis Results to PostgreSQL and PowerBI

This document provides instructions on how to export the graphical analysis results from the ShiftSync model to PostgreSQL and then connect to PowerBI for visualization.

## Prerequisites

1. PostgreSQL database server installed and running
2. PowerBI Desktop installed
3. Python environment with the required dependencies (see `requirements.txt`)

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the PostgreSQL connection details in `Pipeline/Scripts/export_analysis_to_postgres.py`:
   ```python
   DB_HOST = "your_postgres_host"
   DB_PORT = "your_postgres_port"
   DB_NAME = "your_database_name"
   DB_USER = "your_username"
   DB_PASSWORD = "your_password"
   ```

## Running the Export Script

1. Navigate to the project root directory
2. Run the export script:
   ```
   python Pipeline/Scripts/export_analysis_to_postgres.py
   ```

The script will:
1. Load the cleansed data
2. Run the peak hours analysis
3. Prepare the data for PostgreSQL
4. Export the data to PostgreSQL

## Connecting PowerBI to PostgreSQL

1. Open PowerBI Desktop
2. Click "Get Data" > "PostgreSQL database"
3. Enter the server details (host, port, database, username, password)
4. Select "Import" or "DirectQuery" mode
   - Import: Data is loaded into PowerBI and refreshed on demand
   - DirectQuery: Data is queried directly from PostgreSQL in real-time
5. Choose the tables:
   - `peak_hours_analysis`: Contains peak hours identified by each clustering method
   - `overlap_analysis`: Contains the overlap analysis between different clustering methods
   - `peak_hours_summary`: Contains a summary of the most reliable peak hours (high agreement)

## Creating Visualizations in PowerBI

### Recommended Visualizations

1. **Peak Hours by Venue**
   - Create a bar chart showing peak hours for each venue
   - Filter by clustering method (kmeans, gmm, agglo)
   - Use the `peak_hours_analysis` table

2. **Overlap Analysis**
   - Create a heatmap showing the overlap between different clustering methods
   - Use the `overlap_analysis` table
   - Color by `overlap_category` (High, Medium, Low, None)

3. **Summary Dashboard**
   - Create a dashboard showing the most reliable peak hours for each venue
   - Use the `peak_hours_summary` table
   - Filter by `agreement_level` to focus on high agreement hours

4. **Time Series Analysis**
   - Create a line chart showing the normalized order count by hour
   - Highlight peak hours based on the clustering results
   - Use the `overlap_analysis` table

### Feature Selection

In PowerBI, you can create slicers and filters to allow users to:
- Select specific venues (concepts)
- Filter by clustering method
- Filter by agreement level
- Focus on specific time periods

## Troubleshooting

If you encounter issues:

1. **PostgreSQL Connection Issues**
   - Verify your PostgreSQL server is running
   - Check your connection credentials
   - Ensure your PostgreSQL server allows connections from your IP address

2. **Data Export Issues**
   - Check the console output for error messages
   - Verify the data format matches the expected schema
   - Ensure you have the required permissions to create tables and insert data

3. **PowerBI Connection Issues**
   - Verify your PostgreSQL credentials in PowerBI
   - Check if your PostgreSQL server allows connections from PowerBI
   - Try using "Import" mode instead of "DirectQuery" if performance is an issue 