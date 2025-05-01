import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from shapely.geometry import Point
import os
import seaborn as sns
import json


def bandwidth_variability_permutation_test(bandwidth_vector: np.ndarray,
                                           base_gdf: gpd.GeoDataFrame,
                                           n_permutations: int = 999,
                                           alpha: float = 0.05) -> gpd.GeoDataFrame:
    """
    Performs a permutation-based test for local bandwidth variability at each spatial location
    for a single episode.

    Args:
        bandwidth_vector (np.ndarray): 1D array of length n representing local bandwidths.
        base_gdf (gpd.GeoDataFrame): GeoDataFrame with spatial geometries.
        n_permutations (int): Number of permutations to perform.
        alpha (float): Significance level for two-tailed test.

    Returns:
        GeoDataFrame with original bandwidth, p-value, and significance flag.
    """
    n = len(bandwidth_vector)
    print(n)
    print(len(base_gdf))
    assert n == len(
        base_gdf), "Mismatch between bandwidth vector and GeoDataFrame size"

    # Initialize result containers
    p_values = np.zeros(n)

    # Permutation-based p-value computation
    for i in range(n):
        observed = bandwidth_vector[i]
        permuted_values = []

        for _ in range(n_permutations):
            permuted = np.random.permutation(bandwidth_vector)
            permuted_values.append(permuted[i])

        # Two-tailed p-value: proportion of permuted values more extreme than observed
        greater = np.sum(np.abs(permuted_values) >= np.abs(observed))
        p_values[i] = (greater + 1) / \
            (n_permutations + 1)  # add 1 for stability

    # Determine significance
    significant = p_values < alpha

    # Return GeoDataFrame with results
    result_gdf = base_gdf.copy()
    result_gdf['bandwidth'] = bandwidth_vector
    result_gdf['p_value'] = p_values
    result_gdf['significant'] = significant

    return result_gdf


def plot_bandwidth_variability_results_simplified(
    index: int, gdf: gpd.GeoDataFrame, sig_map_dir: str, p_value_map_dir: str
):
    """
    Plots two maps:
    1. Binary map of significant locations (red = significant)
    2. Choropleth map of p-values
    """

    # 1. Significant map
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(ax=ax, color='lightgray', edgecolor='black')
    gdf[gdf['significant']].plot(ax=ax, color='red', edgecolor='black')
    ax.set_title("Significant Local Bandwidth Variability (p < 0.05)")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sig_map_dir, f'significant_map_{index}.png'))

    # 2. P-value choropleth
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(column='p_value', ax=ax, cmap='viridis_r',
             edgecolor='black', legend=True)
    ax.set_title("P-values of Local Bandwidth Variability")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(p_value_map_dir, f'p_value_map_{index}.png'))


if __name__ == "__main__":

    log_dir = r'./locked-logs/georgia/03-super-large-episode/lgwr-2025-04-03-11-02-37-log'
    georgia_shp = gpd.read_file(r'./data/G_utm.shp')

    with open(
        os.path.join(log_dir, 'model_info.json'), 'r'
    ) as file:
        model_info = json.load(file)

        variability_dir = os.path.join(log_dir, 'bandwidth-variability')
        sig_map_dir = os.path.join(variability_dir, 'sig')
        p_value_map_dir = os.path.join(variability_dir, 'p')
        os.makedirs(variability_dir, exist_ok=True)
        os.makedirs(sig_map_dir, exist_ok=True)
        os.makedirs(p_value_map_dir, exist_ok=True)

        for index, episode_record in enumerate(model_info["bandwidth_optimization"]):
            local_bandwidth_vector = np.array(
                json.loads(episode_record['bandwidth'])
            )

            gdf = bandwidth_variability_permutation_test(
                local_bandwidth_vector,
                georgia_shp,
                n_permutations=1000,
                alpha=0.05
            )
            plot_bandwidth_variability_results_simplified(
                index, gdf, sig_map_dir, p_value_map_dir
            )
