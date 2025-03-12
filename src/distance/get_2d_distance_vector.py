import torch
import numpy as np
import numpy.typing as npt
from src.dataset.spatial_dataset import SpatialDataset
from src.distance.haversine_distance.haversine_distance import haversine_distance
from src.distance.euclidean_distance.euclidean_distance import euclidean_distance


def get_2d_distance_vector(index: int, dataset: SpatialDataset) -> npt.NDArray[np.float64]:
    """
    Calculate the distance vector for a specific point in a spatial dataset.

    This function calculates the distance from a target data point (specified by its index)
    to all other points in the dataset. It supports both spherical (using Haversine distance)
    and Cartesian (using Euclidean distance) coordinates.

    Args:
        index (int): The index of the target data point in the dataset.
        dataset (SpatialDataset): The spatial dataset containing all data points.

    Returns:
        npt.NDArray[Any]: A numpy array containing the distances from the target point
                      to every other point in the dataset.

    Raises:
        ValueError: If the dataset has no data points.

    Example:
        >>> dataset = SpatialDataset(data, fieldInfo)
        >>> distances = get_2d_distance_vector(0, dataset)
        >>> print(distances)
        [0.0, 12.4, 23.1, ...]
    """
    if dataset.dataPoints is None:
        raise ValueError(
            "No data points found in the dataset. (get_distance_vector)")

    num_points = len(dataset.dataPoints)
    distances = np.zeros(num_points)

    target_point = dataset.dataPoints[index]
    target_x = target_point.coordinate_x
    target_y = target_point.coordinate_y

    distance_function = haversine_distance if dataset.isSpherical else euclidean_distance

    for i in range(num_points):
        current_point = dataset.dataPoints[i]
        destination_x = current_point.coordinate_x
        destination_y = current_point.coordinate_y

        distances[i] = distance_function(
            target_x, target_y, destination_x, destination_y)
        # if dataset.isSpherical:
        #     # Calculate Haversine distance for spherical data
        #     distances[i] = haversine_distance(
        #         target_y, target_x, destination_y, destination_x)
        # else:
        #     # Calculate Euclidean distance for Cartesian data
        #     distances[i] = euclidean_distance(
        #         target_x, target_y, destination_x, destination_y)

    return distances


def get_2d_distance_vector_torch(index: int,
                                 dataset: SpatialDataset,
                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    """
    Calculate the distance vector for a specific point in a spatial dataset using PyTorch.

    This function calculates the distance from a target data point (specified by its index)
    to all other points in the dataset. It supports both spherical (using Haversine distance)
    and Cartesian (using Euclidean distance) coordinates.

    Args:
        index (int): The index of the target data point in the dataset.
        dataset (SpatialDataset): The spatial dataset containing all data points.
        device (torch.device): The device to store the tensor (CPU/GPU).

    Returns:
        torch.Tensor: A 1D tensor containing the distances from the target point
                      to every other point in the dataset.

    Raises:
        ValueError: If the dataset has no data points.

    Example:
        >>> dataset = SpatialDataset(data, fieldInfo)
        >>> distances = get_2d_distance_vector_torch(0, dataset)
        >>> print(distances)
        tensor([0.0, 12.4, 23.1, ...], device='cuda:0')
    """
    if dataset.dataPoints is None:
        raise ValueError(
            "No data points found in the dataset. (get_distance_vector_torch)")

    num_points = len(dataset.dataPoints)

    distances = torch.zeros(num_points, dtype=torch.float32, device=device)

    target_point = dataset.dataPoints[index]
    target_x = torch.tensor(target_point.coordinate_x,
                            dtype=torch.float32, device=device)
    target_y = torch.tensor(target_point.coordinate_y,
                            dtype=torch.float32, device=device)

    distance_function = haversine_distance if dataset.isSpherical else euclidean_distance

    for i in range(num_points):
        current_point = dataset.dataPoints[i]
        destination_x = torch.tensor(
            current_point.coordinate_x, dtype=torch.float32, device=device)
        destination_y = torch.tensor(
            current_point.coordinate_y, dtype=torch.float32, device=device)

        # ✅ 確保距離計算結果為 PyTorch Tensor
        distance = torch.tensor(
            distance_function(target_x.item(), target_y.item(),
                              destination_x.item(), destination_y.item()),
            dtype=torch.float32, device=device
        )

        distances[i] = distance

    return distances
