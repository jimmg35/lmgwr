import torch
import numpy as np


def euclidean_distance(target_x, target_y, destination_x, destination_y):
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    The Euclidean distance is the straight-line distance between two points
    in a two-dimensional space, given their x and y coordinates.

    Args:
        target_x (float): X-coordinate of the first point.
        target_y (float): Y-coordinate of the first point.
        destination_x (float): X-coordinate of the second point.
        destination_y (float): Y-coordinate of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    Examples:
        >>> euclidean_distance(1.0, 2.0, 4.0, 6.0)
        5.0
    """
    return np.sqrt((destination_x - target_x)**2 + (destination_y - target_y)**2)


def euclidean_distance_torch(
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    destination_x: torch.Tensor,
    destination_y: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Euclidean distance between two points in a 2D plane using PyTorch.

    This function ensures that all operations are performed using PyTorch tensors, 
    allowing for GPU acceleration and enabling autograd for backpropagation.

    Args:
        target_x (torch.Tensor): X-coordinate of the first point.
        target_y (torch.Tensor): Y-coordinate of the first point.
        destination_x (torch.Tensor): X-coordinate of the second point.
        destination_y (torch.Tensor): Y-coordinate of the second point.
        device (torch.device, optional): Device to perform calculations on (CPU/GPU). Defaults to auto-detect.

    Returns:
        torch.Tensor: The Euclidean distance between the two points.

    Examples:
        >>> target_x = torch.tensor(1.0)
        >>> target_y = torch.tensor(2.0)
        >>> destination_x = torch.tensor(4.0)
        >>> destination_y = torch.tensor(6.0)
        >>> distance = euclidean_distance_torch(target_x, target_y, destination_x, destination_y)
        >>> print(distance)  # Expected output: tensor(5.0)
    """

    distance = torch.sqrt((destination_x - target_x) **
                          2 + (destination_y - target_y) ** 2)

    return distance
