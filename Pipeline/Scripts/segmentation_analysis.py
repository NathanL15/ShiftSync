import polars as pl
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from model_segmentation import analyze_peak_hours

# Load your data
data_path = "Data/final_data.csv"
average_hourly_order_counts = pl.read_csv(data_path)

# Group by concept and hour, then calculate the mean normalized order count
concept_hourly_avg = (
    average_hourly_order_counts
    .group_by(['concept', 'hour'])
    .agg(pl.mean('normalized_order_count').alias('normalized_order_count'))
)

concepts = concept_hourly_avg.select('concept').unique().to_series().to_list()

for concept in concepts:
    plt.figure(figsize=(12, 6))
    concept_data = concept_hourly_avg.filter(pl.col('concept') == concept)
    
    # Run the analysis
    analyze_peak_hours(concept_data, kmeans=True, gmm=True, agglo=True, overlap=True, graph=False)

    # Optionally, you can add code to save or display the plots
    # plt.savefig(f"{concept}_analysis.png")
    # plt.show() 