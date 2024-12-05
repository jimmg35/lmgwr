import pytest
from src.distance.euclidean_distance.euclidean_distance import euclidean_distance


def test_distance_between_two_points():
    # Check the distance between two points (0, 0) and (3, 4)
    assert euclidean_distance(0, 0, 3, 4) == 5.0  # 3-4-5 triangle


def test_distance_between_same_point():
    # Check the distance between two points that are the same
    assert euclidean_distance(1, 1, 1, 1) == 0.0


def test_distance_with_negative_coordinates():
    # Check the distance between points with negative coordinates
    assert euclidean_distance(-1, -1, -4, -5) == 5.0  # Another 3-4-5 triangle


def test_distance_from_origin():
    # Check the distance between points where one point is at the origin
    assert euclidean_distance(0, 0, 6, 8) == 10.0  # 6-8-10 triangle


def test_distance_with_floating_point_values():
    # Check the distance between two points with floating-point values
    result = euclidean_distance(1.1, 2.2, 3.3, 4.4)
    # Approximately 3.111 with a tolerance
    expected = pytest.approx(3.111, 0.001)
    assert result == expected
