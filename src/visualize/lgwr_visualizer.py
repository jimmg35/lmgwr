import os
import json
import numpy as np

from src.dataset.spatial_dataset import SpatialDataset
from src.visualize.lgwr_bandwidth_statistics import LgwrBandwidthStatistics
import geopandas as gpd
import matplotlib.pyplot as plt

from libpysal.weights import Queen
from splot.esda import plot_moran
from esda.moran import Moran


class LgwrVisualizer():

    log_path: str
    log_filename: str = 'model_info.json'
    model_info: dict

    dataset: SpatialDataset

    def __init__(self, log_path: str, dataset: SpatialDataset):
        self.log_path = log_path
        self.dataset = dataset
        self.__load_log_json()

    def descriptive_statistics(self, episode: int) -> LgwrBandwidthStatistics:

        optimized_bandwidths_info = self.model_info['bandwidth_optimization']
        info = next(filter(lambda x: x['episode'] ==
                    episode, optimized_bandwidths_info), None)
        if info is None or self.dataset.geometry is None:
            raise ValueError(
                "Episode not found or geometry is None. Please check the episode number or geometry."
            )

        max_bandwidth, min_bandwidth, median_bandwidth, mean_bandwidth, variance_bandwidth, mob_bandwidth = self.basic_statistics(
            episode
        )

        moran_i, moran_p = self.test_spatial_autocorrelation(episode)

        result = LgwrBandwidthStatistics(
            episode=info['episode'],
            reward=info['reward'],
            r2=info['r2'],
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            median_bandwidth=median_bandwidth,
            mean_bandwidth=mean_bandwidth,
            variance_bandwidth=variance_bandwidth,
            mean_absolute_deviation=mob_bandwidth,
            moran_i=moran_i,
            moran_p_value=moran_p
        )
        return result

    def basic_statistics(self, episode: int):
        is_scalar, local_bandwidths = self._get_bandwidth_by_episode(episode)
        if is_scalar:
            max_bandwidth = np.max(local_bandwidths)
            min_bandwidth = np.min(local_bandwidths)
            median_bandwidth = float(np.median(local_bandwidths))
            mean_bandwidth = float(np.mean(local_bandwidths))
            variance_bandwidth = float(np.var(local_bandwidths))
            mob_bandwidth = float(
                np.mean(np.abs(local_bandwidths - mean_bandwidth)))
            return max_bandwidth, min_bandwidth, median_bandwidth, mean_bandwidth, variance_bandwidth, mob_bandwidth

        max_bandwidth = np.max(local_bandwidths)
        min_bandwidth = np.min(local_bandwidths)
        median_bandwidth = float(np.median(local_bandwidths))
        mean_bandwidth = float(np.mean(local_bandwidths))
        variance_bandwidth = float(np.var(local_bandwidths))
        mob_bandwidth = float(
            np.mean(np.abs(local_bandwidths - mean_bandwidth)))
        return max_bandwidth, min_bandwidth, median_bandwidth, mean_bandwidth, variance_bandwidth, mob_bandwidth

    def test_spatial_autocorrelation(self, episode: int, permutations: int = 999) -> tuple[float, float]:
        local_bandwidths = self._get_bandwidth_by_episode(episode)
        w = Queen.from_dataframe(self.dataset.geometry)
        w.transform = 'r'  # type: ignore
        moran = Moran(local_bandwidths, w, permutations=permutations)
        return moran.I, moran.p_sim

    def plot_bandwidth_map_by_episode(self,
                                      episode: int,
                                      result: LgwrBandwidthStatistics,
                                      save: bool = False,
                                      ) -> None:
        if self.dataset.geometry is None:
            return
        local_bandwidths = self._get_bandwidth_by_episode(episode)

        self.dataset.geometry['local_bandwidth'] = local_bandwidths
        self.dataset.geometry.plot(
            column='local_bandwidth',
            cmap='viridis',
            legend=True,
            legend_kwds={'label': "Local Bandwidth"}
        )
        plt.title(f"Bandwidth Map for Episode {episode}")

        textstr = '\n'.join((
            f"Max: {result.max_bandwidth:.2f}",
            f"Min: {result.min_bandwidth:.2f}",
            f"Mean: {result.mean_bandwidth:.2f}",
            f"Median: {result.median_bandwidth:.2f}",
            f"Variance: {result.variance_bandwidth:.2f}",
            f"MAD: {result.mean_absolute_deviation:.2f}",
            f"AICc: {result.reward:.2f}",
            f"R2: {result.r2:.2f}",
            f"Moran's I: {result.moran_i:.3f}",
            f"p-value: {result.moran_p_value:.3f}"
        ))

        plt.gcf().text(0.05, 0.95, textstr, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8),
                       verticalalignment='top')

        if save:
            map_path = os.path.join(self.log_path, 'bandwidth_map')
            if not os.path.exists(map_path):
                os.mkdir(map_path)

            plt.savefig(os.path.join(
                map_path, f"bandwidth_map_episode_{episode}.png"
            ))
        else:
            plt.show()

    def plot_training_process(self, episode: int) -> None:
        try:
            with open(
                os.path.join(
                    self.log_path, f'training_process_{episode}.json'), 'r'
            ) as file:
                training_process = json.load(file)

                aicc_records = json.loads(training_process["aicc_records"])
                r2_records = json.loads(training_process["r2_records"])
                bandwidth_mean_records = json.loads(
                    training_process["bandwidth_mean_records"])
                bandwidth_variance_records = json.loads(
                    training_process["bandwidth_variance_records"])

                fig, axs = plt.subplots(4, 1, figsize=(
                    8, 16), constrained_layout=True)

                # Plot AICc records
                axs[0].plot(range(len(aicc_records)), aicc_records,
                            label='AICc', color='#377BAB')
                axs[0].set_xlabel('Episode')
                axs[0].set_ylabel('AICc')
                axs[0].grid(True)
                axs[0].set_title('AICc over Episodes')

                # Plot R2 records
                axs[1].plot(range(len(r2_records)), r2_records,
                            label='R2', color='#956A88')
                axs[1].set_xlabel('Episode')
                axs[1].set_ylabel('R2')
                axs[1].grid(True)
                axs[1].set_title('R2 over Episodes')

                # Plot Bandwidth Mean records
                axs[2].plot(range(len(bandwidth_mean_records)),
                            bandwidth_mean_records, label='Bandwidth Mean', color='#8EA0CC')
                axs[2].set_xlabel('Episode')
                axs[2].set_ylabel('Bandwidth Mean')
                axs[2].grid(True)
                axs[2].set_title('Bandwidth Mean over Episodes')

                # Plot Bandwidth Variance records
                axs[3].plot(range(len(bandwidth_variance_records)),
                            bandwidth_variance_records, label='Bandwidth Variance', color='#CCB4D7')
                axs[3].set_xlabel('Episode')
                axs[3].set_ylabel('Bandwidth Variance')
                axs[3].grid(True)
                axs[3].set_title('Bandwidth Variance over Episodes')

                # Save the combined figure
                plt.savefig(os.path.join(self.log_path,
                            f'training_process_{episode}_combined.png'))
                plt.close()
        except FileNotFoundError:
            print(f"Training process file for episode {episode} not found.")

    def __load_log_json(self) -> None:
        with open(
            os.path.join(self.log_path, self.log_filename), 'r'
        ) as file:
            self.model_info = json.load(file)

    def _get_bandwidth_by_episode(self, episode: int):
        info = next(filter(lambda x: x['episode'] == episode,
                    self.model_info['bandwidth_optimization']), None)
        if info is None:
            raise ValueError("Episode not found.")

        if isinstance(info['bandwidth'], float):
            return True, info['bandwidth']
        return False, np.array(json.loads(info['bandwidth'])).reshape(-1)
