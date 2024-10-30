
import pytest
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


def test_gwr_kernel(euclidean_dataset):
    gwrKernel = GwrKernel(euclidean_dataset, 100, 'triangular')
    gwrKernel.get_weighted_matrix_by_id(0)
    # distances = get_2d_distance_vector(0, euclidean_dataset)
    # expected_distances = np.array([0., 1414.21, 2828.43,  707.11, 2121.32])
    # np.testing.assert_array_almost_equal(
    #     distances, expected_distances, decimal=2)

# synthetic_data = pd.read_csv(r'../../data/synthetic_dataset.csv')

# spatialDataset = SpatialDataset(
#     synthetic_data,
#     IFieldInfo(
#         predictor_fields=['temperature', 'moisture'],
#         response_field='pm25',
#         coordinate_x_field='coor_x',
#         coordinate_y_field='coor_y'
#     ),
#     isSpherical=True
# )
