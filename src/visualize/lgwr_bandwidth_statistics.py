from dataclasses import dataclass


@dataclass
class LgwrBandwidthStatistics:
    """
    Class to store and display statistics related to the bandwidth of LGWR (Log Writer) in Oracle databases.
    """

    episode: int
    reward: float
    r2: float

    max_bandwidth: float
    min_bandwidth: float
    median_bandwidth: float
    mean_bandwidth: float
    variance_bandwidth: float
    mean_absolute_deviation: float

    moran_i: float
    moran_p_value: float

    def __str__(self) -> str:
        return (
            f"{'-'*40}\n"
            f"Episode: {self.episode} | Reward: {self.reward} | R2: {self.r2}\n"
            f"{'Bandwidth Statistics':^40}\n"
            f"{'-'*40}\n"
            f"{'Statistic':<25} | {'Value':<10}\n"
            f"{'-'*40}\n"
            f"{'Max':<25} | {self.max_bandwidth:.4f}\n"
            f"{'Min':<25} | {self.min_bandwidth:.4f}\n"
            f"{'Median':<25} | {self.median_bandwidth:.4f}\n"
            f"{'Mean':<25} | {self.mean_bandwidth:.4f}\n"
            f"{'Variance':<25} | {self.variance_bandwidth:.4f}\n"
            f"{'Mean Absolute Deviation':<25} | {self.mean_absolute_deviation:.4f}\n"
            f"{'Moran\'s I':<25} | {self.moran_i:.4f}\n"
            f"{'Moran p-value':<25} | {self.moran_p_value:.4f}\n"
            f"{'-'*40}"
        )
