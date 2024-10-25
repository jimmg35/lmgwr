import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    The Euclidean distance is the straight-line distance between two points
    in a two-dimensional space, given their x and y coordinates.

    Args:
        x1 (float): X-coordinate of the first point.
        y1 (float): Y-coordinate of the first point.
        x2 (float): X-coordinate of the second point.
        y2 (float): Y-coordinate of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    Examples:
        >>> euclidean_distance(1.0, 2.0, 4.0, 6.0)
        5.0
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
