
# acknowledgements to Taylor Oshan for the source code
__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, Optional, Literal, TypeAlias
from dataset.spatial_dataset import SpatialDataset
from dataset.interfaces.spatial_dataset import IFieldInfo
from distance.get_2d_distance_vector import get_2d_distance_vector


KernelFunctionType: TypeAlias = Literal['triangular', 'uniform', 'quadratic',
                                        'quartic', 'gaussian', 'bisquare', 'exponential']


class GwrKernel(object):
    """
    GWR kernel function specifications.

    This class builds a spatial weighted matrix for each data point, using a specified
    kernel function to calculate weights based on the distance between data points.

    """
    dataset: SpatialDataset | None = None
    bandwidth: float
    kernel_type: KernelFunctionType = "triangular"

    def __init__(self, dataset: SpatialDataset, bandwidth: float, kernel_type: KernelFunctionType = 'triangular') -> None:
        self.dataset = dataset
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type

    def get_weighted_matrix_by_id(self, index: int) -> None:
        distance_vector = self.__get_distance_vector(index)
        weighted_matrix = self.__calculate_weighted_matrix(
            distance_vector
        )
        print(distance_vector)
        print(weighted_matrix)

    def __get_distance_vector(self, index: int) -> NDArray[np.float64]:
        if self.dataset is None:
            raise ValueError("Dataset is not setup in Kernel")
        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not setup in Kernel")
        return get_2d_distance_vector(index, self.dataset).reshape(-1)

    def __calculate_weighted_matrix(self, distance_vector: NDArray[np.float64]) -> NDArray[np.float64]:
        zs: NDArray[np.float64] = distance_vector / self.bandwidth
        if self.kernel_type == 'triangular':
            return 1 - zs
        elif self.kernel_type == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.kernel_type == 'quadratic':
            return (3. / 4) * (1 - zs**2)
        elif self.kernel_type == 'quartic':
            return (15. / 16) * (1 - zs**2)**2
        elif self.kernel_type == 'gaussian':
            return np.exp(-0.5 * (zs)**2)
        elif self.kernel_type == 'bisquare':
            return (1 - (zs)**2)**2
        elif self.kernel_type == 'exponential':
            return np.exp(-zs)
        raise ValueError('Unsupported kernel function')

    # def __init__(self, i, data, bw=None, fixed=True, function='triangular',
    #                 eps=1.0000001, ids=None, points=None, spherical=False):
    #     """
    #     Initializes the GWR kernel with the specified parameters.

    #     Args:
    #         i (int): Index of the target data point.
    #         data (NDArray): The dataset containing all data points.
    #         bw (float, optional): The bandwidth value for calculating weights.
    #         fixed (bool, optional): If True, a fixed bandwidth is used. If False, adaptive bandwidth is applied.
    #         function (str, optional): The kernel function to use. Default is 'triangular'.
    #         eps (float, optional): A small constant to adjust bandwidth calculation. Default is 1.0000001.
    #         ids (Optional[list], optional): List of data point IDs. Default is None.
    #         points (Optional[NDArray], optional): Precomputed data points for distance calculation. Default is None.
    #         spherical (bool, optional): If True, spherical distance will be used. Default is False.

    #     Raises:
    #         TypeError: If the bandwidth (bw) is not specified.
    #     """
    #     if points is None:
    #         self.dvec = local_cdist(data[i], data, spherical).reshape(-1)
    #     else:
    #         self.dvec = local_cdist(points[i], data, spherical).reshape(-1)

    #     self.function = function.lower()

    #     if bw is None:
    #         raise TypeError('Bandwidth value (bw) must be specified')

    #     if fixed:
    #         self.bandwidth = float(bw)
    #     else:
    #         self.bandwidth = np.partition(
    #             self.dvec,
    #             # partial sort in O(n) Time
    #             int(bw) - 1)[int(bw) - 1] * eps

    #     self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)

    #     if self.function == "bisquare":  # Truncate for bisquare
    #         self.kernel[(self.dvec >= self.bandwidth)] = 0

    # def _kernel_funcs(self, zs):
    #     # functions follow Anselin and Rey (2010) table 5.4
    #     if self.function == 'triangular':
    #         return 1 - zs
    #     elif self.function == 'uniform':
    #         return np.ones(zs.shape) * 0.5
    #     elif self.function == 'quadratic':
    #         return (3. / 4) * (1 - zs**2)
    #     elif self.function == 'quartic':
    #         return (15. / 16) * (1 - zs**2)**2
    #     elif self.function == 'gaussian':
    #         return np.exp(-0.5 * (zs)**2)
    #     elif self.function == 'bisquare':
    #         return (1 - (zs)**2)**2
    #     elif self.function == 'exponential':
    #         return np.exp(-zs)
    #     else:
    #         print('Unsupported kernel function', self.function)

    # def build_spatial_weighted_matrix(self):
    #     """
    #     Build the spatial weighted matrix for a particular data point.

    #     Returns:
    #         NDArray: The spatial weighted matrix for the data point.
    #     """
    #     return self.kernel


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/synthetic_dataset.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['temperature', 'moisture'],
            response_field='pm25',
            coordinate_x_field='coor_x',
            coordinate_y_field='coor_y'
        ),
        isSpherical=True
    )

    gwrKernel = GwrKernel(spatialDataset, 100, 'triangular')
