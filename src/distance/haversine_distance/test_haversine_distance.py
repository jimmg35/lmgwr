import pytest
from src.distance.haversine_distance.haversine_distance import haversine_distance


def test_same_point():
    # Test the same point; the distance should be 0
    assert haversine_distance(0, 0, 0, 0) == 0


def test_known_distance():
    # Test with two known points (e.g., New York City to San Francisco)
    nyc_lat, nyc_lon = 40.7128, -74.0060
    sf_lat, sf_lon = 37.7749, -122.4194
    distance = haversine_distance(nyc_lon, nyc_lat, sf_lon, sf_lat)
    # Expected distance is approximately 4129 km
    assert abs(distance - 4129) < 1


def test_equator_distance():
    # Test two points on the equator with a 90-degree difference in longitude
    # Expected distance is approximately half Earth's circumference
    assert abs(haversine_distance(0, 0, 90, 0) - 10007.5) < 1


def test_poles_distance():
    # Test the distance between the North and South Poles, should be close to twice Earth's radius
    # Expected distance is approximately half Earth's circumference
    assert abs(haversine_distance(0, 90, 0, -90) - 20015) < 1
