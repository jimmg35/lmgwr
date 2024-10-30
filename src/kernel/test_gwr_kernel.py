
import pytest
import numpy as np
import pandas as pd
from dataset.spatial_dataset import SpatialDataset
from dataset.interfaces.spatial_dataset import IFieldInfo
from .gwr_kernel import GwrKernel


@pytest.fixture
def euclidean_dataset():
    # Create a sample dataset for Euclidean distance tests
    data_twd97 = pd.DataFrame({
        'temperature': [22, 24, 19, 21, 23],
        'moisture': [70, 65, 80, 75, 68],
        'pm25': [10, 12, 8, 9, 11],
        'coor_x': [0, 10, 20, 30, 40],
        'coor_y': [0, 0, 0, 0, 0]
    })
    field_info = IFieldInfo(
        predictor_fields=['temperature', 'moisture'],
        response_field='pm25',
        coordinate_x_field='coor_x',
        coordinate_y_field='coor_y'
    )
    dataset = SpatialDataset(
        data=data_twd97,
        fieldInfo=field_info,
        isSpherical=False
    )
    return dataset


def test_gwr_kernel_euclidean_triangular(euclidean_dataset):
    gwrKernel = GwrKernel(euclidean_dataset, 100, 'triangular')
    wi = gwrKernel.get_weighted_matrix_by_id(0)
    expected_wi = np.array([
        [1],
        [0.9],
        [0.8],
        [0.7],
        [0.6]
    ])
    np.testing.assert_array_almost_equal(
        wi, expected_wi, decimal=2)


def test_gwr_kernel_euclidean_uniform(euclidean_dataset):
    gwrKernel = GwrKernel(euclidean_dataset, 100, 'uniform')
    wi = gwrKernel.get_weighted_matrix_by_id(0)
    expected_wi = np.array([
        [0.5],
        [0.5],
        [0.5],
        [0.5],
        [0.5]
    ])
    np.testing.assert_array_almost_equal(
        wi, expected_wi, decimal=2)


def test_gwr_kernel_euclidean_gaussian(euclidean_dataset):
    gwrKernel = GwrKernel(euclidean_dataset, 100, 'gaussian')
    wi = gwrKernel.get_weighted_matrix_by_id(0)
    expected_wi = np.array([
        [1],
        [0.99501248],
        [0.98019867],
        [0.95599748],
        [0.92311635]
    ])
    np.testing.assert_array_almost_equal(
        wi, expected_wi, decimal=2)


def test_gwr_kernel_euclidean_bisquare(euclidean_dataset):
    gwrKernel = GwrKernel(euclidean_dataset, 40, 'bisquare')
    wi = gwrKernel.get_weighted_matrix_by_id(0)
    expected_wi = np.array([
        [1],
        [0.87890625],
        [0.5625],
        [0.19140625],
        [0.]
    ])
    np.testing.assert_array_almost_equal(
        wi, expected_wi, decimal=2)
