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
