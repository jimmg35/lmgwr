from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.cluster.icluster import ICluster


class AgglomerativeCluster(ICluster):
    """
    AgglomerativeCluster class
    This class implements the Agglomerative clustering algorithm.
    It inherits from the ICluster class.
    """

    def __init__(self, log_path: str, n_clusters: int = 5):
        super().__init__(log_path)
        self.n_clusters = n_clusters
        self.agglo = None

    def fit(self):
        data = self.local_bandwidth_vectors
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)

        self.agglo = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.cluster_labels = self.agglo.fit_predict(data_normalized).tolist()

        return self.cluster_labels
