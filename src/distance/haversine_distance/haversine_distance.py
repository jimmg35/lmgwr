import math
import torch


def haversine_distance(target_x, target_y, destination_x, destination_y):
    """
    Calculate the Haversine distance between two points on the Earth's surface.

    The Haversine formula determines the shortest distance over the Earth's surface
    between two points specified by their latitude and longitude. This distance is calculated
    assuming the Earth is a sphere with a radius of 6371 kilometers.

    Args:
        target_x (float): Longitude of the first point in decimal degrees.
        target_y (float): Latitude of the first point in decimal degrees.
        destination_x (float): Longitude of the second point in decimal degrees.
        destination_y (float): Latitude of the second point in decimal degrees.

    Returns:
        float: The distance between the two points in kilometers.

    Examples:
        >>> haversine_distance(-73.935242, 40.730610, -74.0060, 40.7128)
        8.33860776443011
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    target_x_rad = math.radians(target_x)
    target_y_rad = math.radians(target_y)
    destination_x_rad = math.radians(destination_x)
    destination_y_rad = math.radians(destination_y)

    # Calculate the differences in coordinates
    delta_y = destination_y_rad - target_y_rad
    delta_x = destination_x_rad - target_x_rad

    # Apply the Haversine formula
    a = math.sin(delta_y / 2)**2 + math.cos(target_y_rad) * \
        math.cos(destination_y_rad) * math.sin(delta_x / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    return distance


def haversine_distance_torch(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Haversine distance between two points on the Earth (specified in decimal degrees).
    This is the PyTorch version to ensure proper autograd support.

    Args:
        lat1 (torch.Tensor): Latitude of point 1 in degrees.
        lon1 (torch.Tensor): Longitude of point 1 in degrees.
        lat2 (torch.Tensor): Latitude of point 2 in degrees.
        lon2 (torch.Tensor): Longitude of point 2 in degrees.
        device (torch.device): Device to perform calculations on (CPU/GPU).

    Returns:
        torch.Tensor: The Haversine distance between the two points in kilometers.

    Example:
        >>> lat1 = torch.tensor(36.12, dtype=torch.float32)
        >>> lon1 = torch.tensor(-86.67, dtype=torch.float32)
        >>> lat2 = torch.tensor(33.94, dtype=torch.float32)
        >>> lon2 = torch.tensor(-118.40, dtype=torch.float32)
        >>> distance = haversine_distance_torch(lat1, lon1, lat2, lon2)
        >>> print(distance)  # Expected output: ~2887.26 km
    """
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * \
        torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c

    return distance
