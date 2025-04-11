from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

from src.cluster.icluster import ICluster


class SOM(ICluster):
    """
    SOM class
    This class implements the Self-Organizing Map (SOM) algorithm for clustering.
    It inherits from the ICluster class.   
    """

    def __init__(self,
                 log_path: str,
                 som_x: int = 5,
                 som_y: int = 5):
        super().__init__(log_path)
        self.som_x = som_x
        self.som_y = som_y
        self.som = None

    def fit(self):
        data = self.local_bandwidth_vectors

        # Normalize the bandwidth vectors
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)

        # Initialize and train the SOM
        self.som = MiniSom(self.som_x, self.som_y,
                           data.shape[1], sigma=1, learning_rate=0.5)
        self.som.random_weights_init(data_normalized)
        self.som.train_random(data_normalized, 1000)

        # Assign clusters
        self.cluster_labels = []
        for vec in data_normalized:
            winner = self.som.winner(vec)
            cluster_id = winner[0] * self.som_y + \
                winner[1]  # Convert 2D position to 1D label
            self.cluster_labels.append(cluster_id)

        return self.cluster_labels
