def analyze_peak_hours(data, kmeans=True, gmm=True, agglo=True, graph=False):
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
    graph : bool, default=False
        Whether to display graphs of the results
    
    Returns:
    --------
    dict
        Dictionary with concepts as keys and a list of peak hours for each clustering method
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
        
        if graph:
            plt.figure(figsize=(15, 10))
        
        # Track subplot position
        plot_position = 0
        
        # 1. KMeans Clustering
        if kmeans:
            plot_position += 1
            
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
            
            # Plot data if requested
            if graph:
                if kmeans and gmm and agglo:
                    plt.subplot(3, 1, plot_position)
                
                sns.lineplot(
                    data=concept_data_km.to_pandas(),
                    x='hour',
                    y='normalized_order_count'
                )
                
                # Highlight peak hours
                for row in peak_hours.iter_rows(named=True):
                    plt.axvspan(row['hour'] - 0.5, row['hour'] + 0.5, color='yellow', alpha=0.3)
                
                plt.title(f'Average Normalized Order Counts ({concept}) - KMeans Clustering')
                plt.xlabel('Hour of the Day')
                plt.ylabel('Normalized Count')
                plt.grid(True)
                plt.xticks(range(0, 24))
                
                if not (gmm or agglo):
                    plt.tight_layout()
                    plt.show()
        
        # 2. Gaussian Mixture Model
        if gmm:
            plot_position += 1
            
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
            
            # Plot data if requested
            if graph:
                if kmeans and gmm and agglo:
                    plt.subplot(3, 1, plot_position)
                elif kmeans and gmm:
                    plt.subplot(2, 1, plot_position)
                
                sns.lineplot(
                    data=concept_data_gm.to_pandas(),
                    x='hour',
                    y='normalized_order_count'
                )
                
                # Highlight peak hours
                for row in peak_hours.iter_rows(named=True):
                    plt.axvspan(row['hour'] - 0.5, row['hour'] + 0.5, color='yellow', alpha=0.3)
                
                plt.title(f'Average Normalized Order Counts ({concept}) - GMM Clustering')
                plt.xlabel('Hour of the Day')
                plt.ylabel('Normalized Count')
                plt.grid(True)
                plt.xticks(range(0, 24))
                
                if not agglo:
                    plt.tight_layout()
                    plt.show()
        
        # 3. Agglomerative Clustering
        if agglo:
            plot_position += 1
            
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
            
            # Plot data if requested
            if graph:
                if kmeans and gmm and agglo:
                    plt.subplot(3, 1, plot_position)
                elif (kmeans and agglo) or (gmm and agglo):
                    plt.subplot(2, 1, plot_position)
                
                sns.lineplot(
                    data=concept_data_ac.to_pandas(),
                    x='hour',
                    y='normalized_order_count'
                )
                
                # Highlight peak hours
                for row in peak_hours.iter_rows(named=True):
                    plt.axvspan(row['hour'] - 0.5, row['hour'] + 0.5, color='yellow', alpha=0.3)
                
                plt.title(f'Average Normalized Order Counts ({concept}) - Agglomerative Clustering')
                plt.xlabel('Hour of the Day')
                plt.ylabel('Normalized Count')
                plt.grid(True)
                plt.xticks(range(0, 24))
                
                plt.tight_layout()
                plt.show()
        
    return results


# Example usage:
# results = analyze_peak_hours(average_hourly_order_counts, kmeans=True, gmm=True, agglo=True, graph=True)
#
# # To print results:
# for concept, methods in results.items():
#     print(f"Peak hours for {concept}:")
#     for method_name, peak_data in methods.items():
#         print(f"  {method_name.upper()}:")
#         print(f"  {peak_data}")
#     print()