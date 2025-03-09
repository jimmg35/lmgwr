

import numpy as np
import pandas as pd
import pytest
from src.dataset.spatial_dataset import SpatialDataset, IFieldInfo
from src.distance.get_2d_distance_vector import get_2d_distance_vector
from src.log.logger import GwrLogger


@pytest.fixture
def euclidean_dataset():
    # Create a sample dataset for Euclidean distance tests
    data_twd97 = pd.DataFrame({
        'temperature': [22, 24, 19, 21, 23],
        'moisture': [70, 65, 80, 75, 68],
        'pm25': [10, 12, 8, 9, 11],
        'coor_x': [250000, 251000, 252000, 250500, 251500],
        'coor_y': [2650000, 2651000, 2652000, 2650500, 2651500]
    })
    field_info = IFieldInfo(
        predictor_fields=['temperature', 'moisture'],
        response_field='pm25',
        coordinate_x_field='coor_x',
        coordinate_y_field='coor_y'
    )
    logger = GwrLogger()
    dataset = SpatialDataset(
        data_twd97,
        field_info,
        logger,
        isSpherical=False
    )
    return dataset


@pytest.fixture
def spherical_dataset():
    # Create a sample dataset for spherical (Haversine) distance tests
    data_wgs84 = pd.DataFrame({
        'temperature': [22, 24, 19, 21, 23],
        'moisture': [70, 65, 80, 75, 68],
        'pm25': [10, 12, 8, 9, 11],
        'coor_x': [121.5, 121.6, 121.7, 121.55, 121.65],
        'coor_y': [25.0, 25.1, 25.2, 25.05, 25.15]
    })
    field_info = IFieldInfo(
        predictor_fields=['temperature', 'moisture'],
        response_field='pm25',
        coordinate_x_field='coor_x',
        coordinate_y_field='coor_y'
    )
    logger = GwrLogger()
    dataset = SpatialDataset(
        data_wgs84,
        field_info,
        logger,
        isSpherical=True
    )
    return dataset


def test_euclidean_distance(euclidean_dataset):
    distances = get_2d_distance_vector(0, euclidean_dataset)
    expected_distances = np.array([0., 1414.21, 2828.43,  707.11, 2121.32])
    np.testing.assert_array_almost_equal(
        distances, expected_distances, decimal=2)


def test_haversine_distance(spherical_dataset):
    distances = get_2d_distance_vector(0, spherical_dataset)
    expected_distances = np.array([0., 15, 30,  7.5, 22.5])
    np.testing.assert_array_almost_equal(
        distances, expected_distances, decimal=2)
