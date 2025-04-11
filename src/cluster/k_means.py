from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.cluster.icluster import ICluster


class KMeansCluster(ICluster):
    """
    KMeansCluster class
    This class implements the KMeans clustering algorithm.
    It inherits from the ICluster class.
    """

    def __init__(self, log_path: str, n_clusters: int = 5):
        super().__init__(log_path)
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit(self):
        data = self.local_bandwidth_vectors
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(data_normalized)

        self.cluster_labels = self.kmeans.labels_.tolist()
        return self.cluster_labels
