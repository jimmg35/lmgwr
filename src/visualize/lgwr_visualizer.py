import os
import json
import numpy as np

from src.dataset.spatial_dataset import SpatialDataset
import geopandas as gpd
# from pysal.lib import weights
# from pysal.explore import esda
import matplotlib.pyplot as plt


class LgwrVisualizer():

    log_path: str
    log_filename: str = 'model_info.json'
    model_info: dict

    dataset: SpatialDataset

    def __init__(self, log_path: str, dataset: SpatialDataset):
        self.log_path = log_path
        self.dataset = dataset
        self.__load_log_json()

    def summary_bandwidth_info(self) -> None:
        optimized_bandwidths_info = self.model_info['bandwidth_optimization']
        for info in optimized_bandwidths_info:

            local_bandwidths = np.array(
                json.loads(info['bandwidth'])
            ).reshape(-1, 1)

            max_bandwidth = np.max(local_bandwidths)
            min_bandwidth = np.min(local_bandwidths)
            median_bandwidth = np.median(local_bandwidths)
            mean_bandwidth = np.mean(local_bandwidths)
            variance_bandwidth = np.var(local_bandwidths)
            mob_bandwidth = np.mean(np.abs(local_bandwidths - mean_bandwidth))

            print(f"{'-'*40}")
            print(f"Episode: {info['episode']} | Reward: {info['reward']}")
            print(f"{'Bandwidth Statistics':^40}")
            print(f"{'-'*40}")
            print(f"{'Statistic':<20} | {'Value':<20}")
            print(f"{'-'*40}")
            print(f"{'Max':<20} | {max_bandwidth:.4f}")
            print(f"{'Min':<20} | {min_bandwidth:.4f}")
            print(f"{'Median':<20} | {median_bandwidth:.4f}")
            print(f"{'Mean':<20} | {mean_bandwidth:.4f}")
            print(f"{'Variance':<20} | {variance_bandwidth:.4f}")
            print(f"{'Mean Absolute Deviation':<20} | {mob_bandwidth:.4f}")
            print(f"{'-'*40}")

    def plot_bandwidth_map_by_episode(self, episode: int) -> None:
        optimized_bandwidths_info = self.model_info['bandwidth_optimization']
        info = next(filter(lambda x: x['episode'] ==
                    episode, optimized_bandwidths_info), None)
        if info is None or self.dataset.geometry is None:
            return

        # Load the local bandwidth vector of the given episode
        local_bandwidth_vector = np.array(
            json.loads(info['bandwidth'])
        ).reshape(-1, 1)

        # Ensure the length of local_bandwidth_vector matches the length of the geometry
        if len(local_bandwidth_vector) != len(self.dataset.geometry):
            print(
                "The length of local_bandwidth_vector does not match the length of the geometry.")
            return

        # Add the local_bandwidth_vector as a new column to the GeoDataFrame
        self.dataset.geometry['local_bandwidth'] = local_bandwidth_vector

        # Plot the GeoDataFrame with a gradient color based on local_bandwidth
        self.dataset.geometry.plot(
            column='local_bandwidth',
            cmap='viridis',
            legend=True,
            legend_kwds={'label': "Local Bandwidth"}
        )
        plt.title(f"Bandwidth Map for Episode {episode}")
        plt.show()

    # def __calculate_moran_i(self):
    #     if self.dataset.geometry is None:
    #         return

    #     # Ensure the 'local_bandwidth' column exists
    #     if 'local_bandwidth' not in self.dataset.geometry.columns:
    #         print("The 'local_bandwidth' column is missing in the dataset.")
    #         return

    #     # Calculate Moran's I

    #     # Create spatial weights matrix
    #     w = weights.Queen.from_dataframe(self.dataset.geometry)
    #     w.transform = 'r'

    #     # Extract the local_bandwidth column as a numpy array
    #     local_bandwidth = self.dataset.geometry['local_bandwidth'].values

    #     # Compute Moran's I
    #     moran = esda.Moran(local_bandwidth, w)

    #     print(f"Moran's I: {moran.I:.4f}")
    #     print(f"Expected I: {moran.EI:.4f}")
    #     print(f"p-value: {moran.p_sim:.4f}")
    #     print(f"z-score: {moran.z_sim:.4f}")

    def __load_log_json(self) -> None:

        with open(
            os.path.join(self.log_path, self.log_filename), 'r'
        ) as file:
            self.model_info = json.load(file)
