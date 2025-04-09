def analyze_peak_hours(data, kmeans=True, gmm=True, agglo=True, overlap=True, graph=False):
    """
    Analyze order data to identify peak hours using different clustering methods.
    
    Parameters:
    -----------
    data : polars.DataFrame
        DataFrame containing 'concept', 'hour', and 'normalized_order_count' columns
    kmeans : bool, default=True
        Whether to run KMeans clustering
    gmm : bool, default=True
        Whether to run Gaussian Mixture Model clustering
    agglo : bool, default=True
        Whether to run Agglomerative clustering
    overlap : bool, default=True
        Whether to analyze and visualize the overlap between different clustering results
    graph : bool, default=False
        Whether to display the overlap graph
    
    Returns:
    --------
    dict
        Dictionary with concepts as keys and a list of peak hours for each clustering method,
        plus 'overlap_analysis' if overlap=True
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt
    import seaborn as sns
    import polars as pl
    import numpy as np
    
    # Group by concept and hour, then calculate the mean normalized order count
    concept_hourly_avg = (
        data
        .group_by(['concept', 'hour'])
        .agg(pl.mean('normalized_order_count').alias('normalized_order_count'))
    )

    concepts = concept_hourly_avg.select('concept').unique().to_series().to_list()
    
    # Store results
    results = {}
    
    for concept in concepts:
        results[concept] = {}
        concept_data = concept_hourly_avg.filter(pl.col('concept') == concept)
        
        # Reshape data for clustering
        X = concept_data.select('normalized_order_count').to_numpy()
        
        # Initialize dictionaries to track peak hours by method
        peak_hours_by_method = {}
        
        # 1. KMeans Clustering
        if kmeans:
            # Apply KMeans clustering (choosing 3 clusters for peak, mid, low)
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            km_labels = km.fit_predict(X)
            
            concept_data_km = concept_data.with_columns(
                pl.Series(name='cluster', values=km_labels)
            )
            
            # Determine which cluster represents peak hours (highest avg order count)
            cluster_means = (
                concept_data_km
                .group_by('cluster')
                .agg(pl.mean('normalized_order_count').alias('cluster_mean'))
                .sort('cluster_mean', descending=True)
            )
            peak_cluster = cluster_means[0, 'cluster']
            
            # Identify peak hours
            peak_hours = concept_data_km.filter(pl.col('cluster') == peak_cluster)
            
            # Store results
            results[concept]['kmeans'] = peak_hours.select(['hour', 'normalized_order_count']).sort('hour')
            peak_hours_by_method['kmeans'] = set(peak_hours.select('hour').to_series().to_list())
        
        # 2. Gaussian Mixture Model
        if gmm:
            # Apply Gaussian Mixture Model
            gm = GaussianMixture(n_components=3, random_state=42)
            gm_labels = gm.fit_predict(X)
            
            concept_data_gm = concept_data.with_columns(
                pl.Series(name='cluster', values=gm_labels)
            )
            
            # Determine peak cluster
            cluster_means = (
                concept_data_gm
                .group_by('cluster')
                .agg(pl.mean('normalized_order_count').alias('cluster_mean'))
                .sort('cluster_mean', descending=True)
            )
            peak_cluster = cluster_means[0, 'cluster']
            
            # Identify peak hours
            peak_hours = concept_data_gm.filter(pl.col('cluster') == peak_cluster)
            
            # Store results
            results[concept]['gmm'] = peak_hours.select(['hour', 'normalized_order_count']).sort('hour')
            peak_hours_by_method['gmm'] = set(peak_hours.select('hour').to_series().to_list())
        
        # 3. Agglomerative Clustering
        if agglo:
            # Apply Agglomerative Clustering
            ac = AgglomerativeClustering(n_clusters=3)
            ac_labels = ac.fit_predict(X)
            
            concept_data_ac = concept_data.with_columns(
                pl.Series(name='cluster', values=ac_labels)
            )
            
            # Determine peak cluster
            cluster_means = (
                concept_data_ac
                .group_by('cluster')
                .agg(pl.mean('normalized_order_count').alias('cluster_mean'))
                .sort('cluster_mean', descending=True)
            )
            peak_cluster = cluster_means[0, 'cluster']
            
            # Identify peak hours
            peak_hours = concept_data_ac.filter(pl.col('cluster') == peak_cluster)
            
            # Store results
            results[concept]['agglo'] = peak_hours.select(['hour', 'normalized_order_count']).sort('hour')
            peak_hours_by_method['agglo'] = set(peak_hours.select('hour').to_series().to_list())
        
        # Analyze overlap between methods if requested
        if overlap and len(peak_hours_by_method) > 0:
            # Create a DataFrame with all 24 hours
            all_hours = pl.DataFrame({'hour': range(24)})
            
            # Count how many methods consider each hour a peak hour
            overlap_counts = {}
            for hour in range(24):
                count = sum(1 for method, hours in peak_hours_by_method.items() if hour in hours)
                overlap_counts[hour] = count
            
            # Create a DataFrame with overlap information
            overlap_df = (
                all_hours
                .with_columns(
                    pl.Series(name='overlap_count', values=[overlap_counts[h] for h in range(24)])
                )
                .join(
                    concept_data.select(['hour', 'normalized_order_count']),
                    on='hour',
                    how='left'
                )
            )
            
            # Add categories for coloring based on overlap count
            overlap_df = overlap_df.with_columns(
                pl.when(pl.col('overlap_count') == 3).then(pl.lit('High (3)'))
                .when(pl.col('overlap_count') == 2).then(pl.lit('Medium (2)'))
                .when(pl.col('overlap_count') == 1).then(pl.lit('Low (1)'))
                .otherwise(pl.lit('None (0)'))
                .alias('overlap_category')
            )
            
            # Store overlap analysis in results
            results[concept]['overlap_analysis'] = overlap_df
            
            # Only display overlap graph if graph=True
            if graph:
                plt.figure(figsize=(15, 8))
                
                # Create a color palette for the overlap categories
                colors = {'High (3)': 'red', 'Medium (2)': 'yellow', 'Low (1)': 'green', 'None (0)': 'lightgray'}
                
                # Plot normalized order counts
                sns.lineplot(
                    data=concept_data.to_pandas(),
                    x='hour',
                    y='normalized_order_count',
                    marker='o'
                )
                
                # Highlight hours based on overlap count
                for hour in range(24):
                    if overlap_counts[hour] > 0:
                        color = colors[overlap_df.filter(pl.col('hour') == hour)[0, 'overlap_category']]
                        plt.axvspan(hour - 0.5, hour + 0.5, color=color, alpha=0.3)
                
                # Add a legend for the overlap categories
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', alpha=0.3, label='High agreement (3 methods)'),
                    Patch(facecolor='yellow', alpha=0.3, label='Medium agreement (2 methods)'),
                    Patch(facecolor='green', alpha=0.3, label='Low agreement (1 method)'),
                ]
                plt.legend(handles=legend_elements)
                
                plt.title(f'Peak Hour Overlap Analysis for {concept}')
                plt.xlabel('Hour of the Day')
                plt.ylabel('Normalized Order Count')
                plt.grid(True)
                plt.xticks(range(0, 24))
                plt.tight_layout()
                plt.show()
    
    return results