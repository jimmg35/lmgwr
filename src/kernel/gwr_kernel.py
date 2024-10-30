
# acknowledgements to Taylor Oshan for the source code
__author__ = "Taylor Oshan tayoshan@gmail.com"

import numpy as np
import pandas as pd
import numpy.typing as npt
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

    Attributes:
        dataset (SpatialDataset | None): The spatial dataset used to calculate weights.
        bandwidth (float): The bandwidth parameter controlling the kernel's spatial influence.
        kernel_type (KernelFunctionType): The type of kernel function to use for weight calculations.
    """
    dataset: SpatialDataset | None = None
    bandwidth: float
    kernel_type: KernelFunctionType = "triangular"

    def __init__(self, dataset: SpatialDataset, bandwidth: float, kernel_type: KernelFunctionType = 'triangular') -> None:
        """
        Initializes the GwrKernel with a dataset, bandwidth, and kernel type.

        Args:
            dataset (SpatialDataset): The spatial dataset for generating the weighted matrix.
            bandwidth (float): The bandwidth parameter for controlling spatial influence.
            kernel_type (KernelFunctionType, optional): The kernel function type to use; 
                defaults to 'triangular'.
        """
        self.dataset = dataset
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type

    def get_weighted_matrix_by_id(self, index: int) -> npt.NDArray[np.float64]:
        """
        Computes the weighted matrix for a specific data point by index.

        This function retrieves the distance vector for the specified data point index, 
        then calculates the weights using the chosen kernel function.

        Args:
            index (int): The index of the data point for which to compute weights.

        Returns:
            npt.NDArray[np.float64]: A 2D array representing the weighted matrix for 
                the specified data point.
        """
        distance_vector = self.__get_distance_vector(index)
        weighted_matrix = self.__calculate_weighted_matrix(
            distance_vector
        )
        if self.kernel_type == 'bisquare':
            weighted_matrix[(distance_vector >= self.bandwidth)] = 0
        return weighted_matrix.reshape(-1, 1)

    def __get_distance_vector(self, index: int) -> npt.NDArray[np.float64]:
        """
        Retrieves the distance vector for a specific data point index.

        This function calculates the distances from a specified data point to all other points 
        in the dataset.

        Args:
            index (int): The index of the data point to calculate distances from.

        Returns:
            npt.NDArray[np.float64]: A 1D array of distances from the specified data point 
                to all other points.

        Raises:
            ValueError: If the dataset or data points are not initialized.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not setup in Kernel")
        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not setup in Kernel")
        return get_2d_distance_vector(index, self.dataset).reshape(-1)

    def __calculate_weighted_matrix(self, distance_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculates the weighted matrix based on the kernel function and distance vector.

        This function applies the specified kernel function to the normalized distance vector 
        to calculate weights.

        Args:
            distance_vector (npt.NDArray[np.float64]): A 1D array of distances to apply 
                the kernel function.

        Returns:
            npt.NDArray[np.float64]: A 1D array of weights calculated from the distance vector.

        Raises:
            ValueError: If the kernel function type is unsupported.
        """
        zs: npt.NDArray[np.float64] = distance_vector / self.bandwidth
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
