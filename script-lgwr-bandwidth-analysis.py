import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from src.cluster.som import SOM
from src.cluster.k_means import KMeansCluster
from src.cluster.agglomerative import AgglomerativeCluster
from src.cluster.dbscan import DBSCANCluster

if __name__ == "__main__":

    log_path = "./locked-logs/georgia/03-super-large-episode/lgwr-2025-04-03-11-02-37-log"

    som = SOM(
        log_path=log_path,
        som_x=5,
        som_y=5
    )

    k_means = KMeansCluster(
        log_path=log_path,
        n_clusters=5
    )

    agglomerative = AgglomerativeCluster(
        log_path=log_path,
        n_clusters=5
    )

    dbscan = DBSCANCluster(
        log_path=log_path,
        eps=0.5,
        min_samples=5
    )

    labels_som = som.fit()
    labels_kmeans = k_means.fit()
    labels_agglomerative = agglomerative.fit()
    labels_dbscan = dbscan.fit()

    print("SOM vs KMeans - ARI:", adjusted_rand_score(labels_som, labels_kmeans))
    print("SOM vs Agglomerative - NMI:",
          normalized_mutual_info_score(labels_som, labels_agglomerative))
    print("SOM vs DBSCAN - FMI:", fowlkes_mallows_score(labels_som, labels_dbscan))

    def plot_confusion_matrix(labels1, labels2, title="Cluster Overlap"):
        df = pd.DataFrame({"Method1": labels1, "Method2": labels2})
        confusion = pd.crosstab(df["Method1"], df["Method2"])

        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Method2 Clusters")
        plt.ylabel("Method1 Clusters")
        plt.show()

    plot_confusion_matrix(labels_som, labels_kmeans, title="SOM vs KMeans")
    plot_confusion_matrix(labels_som, labels_agglomerative,
                          title="SOM vs Agglomerative")
    plot_confusion_matrix(labels_som, labels_dbscan, title="SOM vs DBSCAN")
