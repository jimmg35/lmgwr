import torch
import numpy as np
import numpy.typing as npt
from src.dataset.spatial_dataset import SpatialDataset
from src.distance.haversine_distance.haversine_distance import haversine_distance, haversine_distance_torch
from src.distance.euclidean_distance.euclidean_distance import euclidean_distance, euclidean_distance_torch


def calculate_distance_vector_by_id(index: int, dataset: SpatialDataset) -> npt.NDArray[np.float64]:
    """
    Calculate the distance vector for a given index in a dataset.

    Args:
        index (int): The index of the point for which to calculate the distance vector.
        dataset (SpatialDataset): The dataset containing the points.

    Returns:
        npt.NDArray[np.float64]: A vector of distances from the point at the given index to all other points in the dataset.
    """

    target_point = dataset.coordinates[index]
    distance_function = haversine_distance if dataset.isSpherical else euclidean_distance

    distances = np.array([
        distance_function(target_point[0], target_point[1], point[0], point[1])
        for point in dataset.coordinates
    ])

    return distances
