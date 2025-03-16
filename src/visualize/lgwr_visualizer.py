import os
import json
import numpy as np


class LgwrVisualizer():

    log_path: str
    log_filename: str = 'model_info.json'
    model_info: dict

    def __init__(self, log_path: str):
        self.log_path = log_path
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
        if info is None:
            return

        local_bandwidths = np.array(
            json.loads(info['bandwidth'])
        ).reshape(-1, 1)

        print(local_bandwidths)

    def __load_log_json(self) -> None:

        with open(
            os.path.join(self.log_path, self.log_filename), 'r'
        ) as file:
            self.model_info = json.load(file)
