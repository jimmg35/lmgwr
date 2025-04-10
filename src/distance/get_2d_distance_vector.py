import torch
import numpy as np
import numpy.typing as npt
from src.dataset.spatial_dataset import SpatialDataset
from src.distance.haversine_distance.haversine_distance import haversine_distance, haversine_distance_torch
from src.distance.euclidean_distance.euclidean_distance import euclidean_distance, euclidean_distance_torch


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

    # num_points = len(dataset.dataPoints)
    distances = np.zeros(len(dataset))

    target_point = dataset.coordinates[index]
    target_x = target_point[0]
    target_y = target_point[1]

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

    # ✅ 移除 requires_grad=True，這個 Tensor 會在 forward 計算過程中需要梯度
    distances = torch.zeros(num_points, dtype=torch.float32,
                            device=device, requires_grad=True)

    target_point = dataset.dataPoints[index]
    target_x = torch.tensor(target_point.coordinate_x,
                            dtype=torch.float32, device=device, requires_grad=True)
    target_y = torch.tensor(target_point.coordinate_y,
                            dtype=torch.float32, device=device, requires_grad=True)

    distance_function = haversine_distance_torch if dataset.isSpherical else euclidean_distance_torch

    # ✅ 改為 batch 運算來避免 in-place operation
    all_x = torch.tensor(
        [p.coordinate_x for p in dataset.dataPoints], dtype=torch.float32, device=device, requires_grad=True)
    all_y = torch.tensor(
        [p.coordinate_y for p in dataset.dataPoints], dtype=torch.float32, device=device, requires_grad=True)

    distances = distance_function(target_x, target_y, all_x, all_y)

    return distances
