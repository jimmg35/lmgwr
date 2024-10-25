import pytest
from haversine_distance import haversine_distance


def test_haversine_distance():
    # Test case 1: Distance between the same points should be 0
    assert haversine_distance(0, 0, 0, 0) == 0.0

    # Test case 2: Distance between known points (New York to Los Angeles)
    # New York (40.7128° N, 74.0060° W) and Los Angeles (34.0522° N, 118.2437° W)
    ny_lat, ny_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437
    # Approximate distance in kilometers
    expected_distance = pytest.approx(3936.98, 0.1)
    assert haversine_distance(ny_lat, ny_lon, la_lat,
                              la_lon) == expected_distance

    # Test case 3: Check distance between points near the equator
    assert haversine_distance(0, 0, 0, 1) == pytest.approx(
        111.32, 0.1)  # 1 degree longitude ~ 111.32 km at the equator

    # Test case 4: Check distance between the North Pole and the Equator
    north_pole_lat, north_pole_lon = 90, 0
    equator_lat, equator_lon = 0, 0
    # Half the Earth's circumference in km
    expected_distance = pytest.approx(10007.54, 0.1)
    assert haversine_distance(
        north_pole_lat, north_pole_lon, equator_lat, equator_lon) == expected_distance

    # Test case 5: Check distance between antipodal points (opposite sides of the globe)
    # Earth's diameter in kilometers
    assert haversine_distance(0, 0, -0, 180) == pytest.approx(20015.08, 0.1)
