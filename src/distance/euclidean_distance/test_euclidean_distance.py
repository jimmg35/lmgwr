import pytest
from euclidean_distance import euclidean_distance


def test_euclidean_distance():
    # Test case 1: Check the distance between two points (0, 0) and (3, 4)
    assert euclidean_distance(0, 0, 3, 4) == 5.0  # 3-4-5 triangle

    # Test case 2: Check the distance between two points that are the same
    assert euclidean_distance(1, 1, 1, 1) == 0.0

    # Test case 3: Check the distance between points with negative coordinates
    assert euclidean_distance(-1, -1, -4, -5) == 5.0  # Another 3-4-5 triangle

    # Test case 4: Check the distance between points where one point is at the origin
    assert euclidean_distance(0, 0, 6, 8) == 10.0  # 6-8-10 triangle

    # Test case 5: Check the distance between two points with floating-point values
    result = euclidean_distance(1.1, 2.2, 3.3, 4.4)
    # Approximately 3.111 with a tolerance
    expected = pytest.approx(3.111, 0.001)
    assert result == expected
