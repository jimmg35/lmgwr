
import pytest
import numpy as np
import pandas as pd
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.kernel.gwr_kernel import GwrKernel
from src.log.logger import GwrLogger


# @pytest.fixture
# def euclidean_dataset():
#     # Create a sample dataset for Euclidean distance tests
#     data_twd97 = pd.DataFrame({
#         'temperature': [22, 24, 19, 21, 23],
#         'moisture': [70, 65, 80, 75, 68],
#         'pm25': [10, 12, 8, 9, 11],
#         'coor_x': [0, 10, 20, 30, 40],
#         'coor_y': [0, 0, 0, 0, 0]
#     })
#     field_info = IFieldInfo(
#         predictor_fields=['temperature', 'moisture'],
#         response_field='pm25',
#         coordinate_x_field='coor_x',
#         coordinate_y_field='coor_y'
#     )
#     logger = GwrLogger()
#     dataset = SpatialDataset(
#         data_twd97,
#         field_info,
#         logger,
#         isSpherical=False
#     )
#     return dataset


# def test_gwr_kernel_euclidean_triangular(euclidean_dataset):
#     logger = GwrLogger()
#     gwrKernel = GwrKernel(euclidean_dataset, logger, 'triangular')
#     gwrKernel.update_bandwidth(100)
#     wi = gwrKernel.get_weighted_matrix_by_id(0)

#     print(wi)
#     expected_wi = np.array([
#         [1],
#         [0.9],
#         [0.8],
#         [0.7],
#         [0.6]
#     ])
#     np.testing.assert_array_almost_equal(
#         wi, expected_wi, decimal=2)


# def test_gwr_kernel_euclidean_uniform(euclidean_dataset):
#     logger = GwrLogger()
#     gwrKernel = GwrKernel(euclidean_dataset, logger, 'uniform')
#     gwrKernel.update_bandwidth(100)
#     wi = gwrKernel.get_weighted_matrix_by_id(0)
#     expected_wi = np.array([
#         [0.5],
#         [0.5],
#         [0.5],
#         [0.5],
#         [0.5]
#     ])
#     np.testing.assert_array_almost_equal(
#         wi, expected_wi, decimal=2)


# def test_gwr_kernel_euclidean_gaussian(euclidean_dataset):
#     logger = GwrLogger()
#     gwrKernel = GwrKernel(euclidean_dataset, logger, 'gaussian')
#     gwrKernel.update_bandwidth(100)
#     wi = gwrKernel.get_weighted_matrix_by_id(0)
#     expected_wi = np.array([
#         [1],
#         [0.99501248],
#         [0.98019867],
#         [0.95599748],
#         [0.92311635]
#     ])
#     np.testing.assert_array_almost_equal(
#         wi, expected_wi, decimal=2)


# def test_gwr_kernel_euclidean_bisquare(euclidean_dataset):
#     logger = GwrLogger()
#     gwrKernel = GwrKernel(euclidean_dataset, logger, 'bisquare')
#     gwrKernel.update_bandwidth(40)
#     wi = gwrKernel.get_weighted_matrix_by_id(0)
#     expected_wi = np.array([
#         [1],
#         [0.87890625],
#         [0.5625],
#         [0.19140625],
#         [0.]
#     ])
#     np.testing.assert_array_almost_equal(
#         wi, expected_wi, decimal=2)
