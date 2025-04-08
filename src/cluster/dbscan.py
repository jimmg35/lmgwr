from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.cluster.icluster import ICluster


class DBSCANCluster(ICluster):
    """
    DBSCANCluster class
    This class implements the DBSCAN clustering algorithm.
    It inherits from the ICluster class.
    """

    def __init__(self, log_path: str, eps: float = 0.5, min_samples: int = 5):
        super().__init__(log_path)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None

    def fit(self):
        data = self.local_bandwidth_vectors
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = self.dbscan.fit_predict(data_normalized).tolist()

        return self.cluster_labels
