import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from shapely.geometry import Point
import os
import seaborn as sns
import json


def analyze_bandwidth_variability(bandwidth_array: np.ndarray, base_gdf: gpd.GeoDataFrame):
    """
    Given a 2D numpy array of shape (T, n), and a base GeoDataFrame with geometries,
    this function:

    1. Computes variance of bandwidths at each location.
    2. Calculates Z-scores and identifies significant variability.
    3. Plots the significant locations using real spatial geometry.

    Returns:
        GeoDataFrame with variance, z-score, significance.
    """
    T, n = bandwidth_array.shape

    assert n == len(
        base_gdf), "Mismatch between number of locations and geometry"

    # Step 1: Variance
    bandwidth_variance = np.var(bandwidth_array, axis=0)

    # Step 2: Z-score
    z_scores = zscore(bandwidth_variance)
    significance_mask = np.abs(z_scores) > 1.96

    # Step 3: Add to real geometry
    gdf = base_gdf.copy()
    gdf['variance'] = bandwidth_variance
    gdf['z_score'] = z_scores
    gdf['significant'] = significance_mask

    return gdf


def plot_bandwidth_variability_results_separately(gdf: gpd.GeoDataFrame):
    """
    Plots each of the following in a separate figure:
    1. Binary map of significant locations
    2. Z-score choropleth
    3. Variance choropleth
    4. Histogram of Z-scores
    """

    # 1. Significant map
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(ax=ax, color='lightgray', edgecolor='black')
    gdf[gdf['significant']].plot(ax=ax, color='red', edgecolor='black')
    ax.set_title("Significant Local Bandwidth Variability (|Z| > 1.96)")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 2. Z-score map
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(column='z_score', ax=ax, cmap='coolwarm',
             edgecolor='black', legend=True)
    ax.set_title("Z-score of Bandwidth Variance")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 3. Variance map
    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(column='variance', ax=ax, cmap='YlGnBu',
             edgecolor='black', legend=True)
    ax.set_title("Bandwidth Variance")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # 4. Histogram of Z-scores
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(gdf['z_score'],  # type: ignore
                 bins=30, ax=ax, color='skyblue', kde=True)
    ax.axvline(x=1.96, color='red', linestyle='--', label='z = ±1.96')
    ax.axvline(x=-1.96, color='red', linestyle='--')
    ax.set_title("Histogram of Z-scores")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    log_dir = r'./locked-logs/georgia/03-super-large-episode/lgwr-2025-04-03-11-02-37-log/model_info.json'
    georgia_shp = gpd.read_file(r'./data/G_utm.shp')

    with open(
        os.path.join(log_dir), 'r'
    ) as file:
        model_info = json.load(file)

        mat = []
        for episode_record in model_info["bandwidth_optimization"]:
            local_bandwidth_vector = np.array(
                json.loads(episode_record['bandwidth'])
            )

            mat.append(local_bandwidth_vector)
        mat = np.array(mat)

        gdf = analyze_bandwidth_variability(mat, georgia_shp)
        plot_bandwidth_variability_results_separately(gdf)
