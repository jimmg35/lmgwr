
import numpy as np
import numpy.typing as npt
from typing import Literal, TypeAlias, Dict

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.distance.get_2d_distance_vector import get_2d_distance_vector
from src.log.gwr_logger import GwrLogger
from src.log.ilogger import ILogger

KernelFunctionType: TypeAlias = Literal['triangular', 'uniform', 'quadratic',
                                        'quartic', 'gaussian', 'bisquare', 'exponential']
KernelBandwidthType: TypeAlias = Literal['distance_based', 'adaptive']


# This is an interface class for kernel
class IKernel:

    logger: ILogger
    dataset: SpatialDataset | None = None
    bandwidth: float | None = None
    kernel_type: KernelFunctionType = "bisquare"
    kernel_bandwidth_type: KernelBandwidthType = "adaptive"
    weighted_matrix_cache: Dict[int, npt.NDArray[np.float64]] = {}
    distance_vector_cache: Dict[int, npt.NDArray[np.float64]] = {}

    def __init__(self,
                 dataset: SpatialDataset,
                 logger: ILogger,
                 kernel_type: KernelFunctionType = 'bisquare',
                 kernel_bandwidth_type: KernelBandwidthType = 'adaptive'
                 ) -> None:
        """
        Initializes the GwrKernel with a dataset, bandwidth, and kernel type.

        Args:
            dataset (SpatialDataset): The spatial dataset for generating the weighted matrix.
            bandwidth (float): The bandwidth parameter for controlling spatial influence.
            kernel_type (KernelFunctionType, optional): The kernel function type to use; 
                defaults to 'triangular'.
        """
        self.dataset = dataset
        self.logger = logger
        self.kernel_type = kernel_type
        self.kernel_bandwidth_type = kernel_bandwidth_type
        self.logger.append_info(
            f"{self.__class__.__name__} : Kernel is initialized.")

    def update_bandwidth(self, bandwidth: float) -> None:
        """
            for gwr
        """
        raise NotImplementedError("Method not implemented")

    def update_local_bandwidth(self, index: int, bandwidth: float):
        """
            for lgwr
        """
        raise NotImplementedError("Method not implemented")

    def get_weighted_matrix_by_id(self, index: int) -> npt.NDArray[np.float64]:
        """
        Returns the weighted matrix for all data points.

        This function retrieves the weighted matrix for all data points in the dataset.

        Returns:
            Dict[int, npt.NDArray[np.float64]]: A dictionary of weighted matrices for each data point.
        """
        if index not in self.weighted_matrix_cache:
            raise ValueError(
                f"Weighted matrix for index {index} is not found in the cache, please update the bandwidth first")
        return self.weighted_matrix_cache[index]

    def get_distance_vector_by_id(self, index: int) -> npt.NDArray[np.float64]:
        """
        Returns the distance vector for a specific data point by index.

        This function retrieves the distance vector for the specified data point index.

        Args:
            index (int): The index of the data point for which to retrieve the distance vector.

        Returns:
            npt.NDArray[np.float64]: A 1D array of distances from the specified data point to all other points.
        """
        if index not in self.distance_vector_cache:
            raise ValueError(
                f"Distance vector for index {index} is not found in the cache, please update the bandwidth first")
        return self.distance_vector_cache[index]

    def update_weighted_matrix_by_id(self, index: int) -> None:
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

        # retrive the distance vector from cache if found,
        # or initialize it.
        distance_vector = self.__calculate_distance_vector(index)
        self.__calculate_weighted_matrix(
            index,
            distance_vector
        )

    def __calculate_distance_vector(self, index: int) -> npt.NDArray[np.float64]:
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

        # retrieve the distance vector from the cache if it exists
        if index in self.distance_vector_cache:
            return self.distance_vector_cache[index]

        # or calculate and store the distance vector in the cache
        distance_vector_i = get_2d_distance_vector(
            index,
            self.dataset
        ).reshape(-1)
        self.distance_vector_cache[index] = distance_vector_i

        return distance_vector_i

    def __calculate_weighted_matrix(self,
                                    index: int,
                                    distance_vector: npt.NDArray[np.float64],
                                    eps=1.0000001):
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
        if self.bandwidth is None:
            raise ValueError("Bandwidth is not set up in Kernel")

        weighted_matrix: npt.NDArray[np.float64] = np.zeros(
            distance_vector.shape)

        distnace_bandwidth = self.bandwidth
        if self.kernel_bandwidth_type == 'adaptive':
            # partial sort in O(n) Time
            distnace_bandwidth = np.partition(
                distance_vector,
                int(self.bandwidth) - 1
            )[int(self.bandwidth) - 1] * eps

        zs: npt.NDArray[np.float64] = distance_vector / distnace_bandwidth
        if self.kernel_type == 'triangular':
            weighted_matrix = 1 - zs
        elif self.kernel_type == 'uniform':
            weighted_matrix = np.ones(zs.shape) * 0.5
        elif self.kernel_type == 'quadratic':
            weighted_matrix = (3. / 4) * (1 - zs**2)
        elif self.kernel_type == 'quartic':
            weighted_matrix = (15. / 16) * (1 - zs**2)**2
        elif self.kernel_type == 'gaussian':
            weighted_matrix = np.exp(-0.5 * (zs)**2)
        elif self.kernel_type == 'bisquare':
            weighted_matrix = (1 - (zs)**2)**2
        elif self.kernel_type == 'exponential':
            weighted_matrix = np.exp(-zs)
        else:
            raise ValueError('Unsupported kernel function')

        if self.kernel_type == 'bisquare':
            weighted_matrix[(distance_vector >= distnace_bandwidth)] = 0

        # store the weighted matrix in the cache
        self.weighted_matrix_cache[index] = weighted_matrix.reshape(-1, 1)
