import math


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
    delta_x = destination_x_rad - target_x_rad
    delta_y = destination_y_rad - target_y_rad

    # Apply the Haversine formula
    a = math.sin(delta_y / 2)**2 + math.cos(target_y_rad) * \
        math.cos(destination_y_rad) * math.sin(delta_x / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    return distance
