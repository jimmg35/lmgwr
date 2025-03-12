
import numpy as np
import numpy.typing as npt
from typing import Literal, TypeAlias, Dict
from torch import Tensor
import torch

from src.dataset.spatial_dataset import SpatialDataset
from src.distance.get_2d_distance_vector import get_2d_distance_vector, get_2d_distance_vector_torch
from src.log.ilogger import ILogger

KernelFunctionType: TypeAlias = Literal['triangular', 'uniform', 'quadratic',
                                        'quartic', 'gaussian', 'bisquare', 'exponential']
KernelBandwidthType: TypeAlias = Literal['distance_based', 'adaptive']


# This is an interface class for kernel
class IKernel:

    logger: ILogger
    dataset: SpatialDataset | None = None
    bandwidth: float | Tensor | None = None
    optimizeMode: Literal['cuda', 'cpu'] = 'cuda'
    kernel_type: KernelFunctionType = "bisquare"
    kernel_bandwidth_type: KernelBandwidthType = "adaptive"
    weighted_matrix_cache: Dict[int, npt.NDArray[np.float64] | Tensor] = {}
    distance_vector_cache: Dict[int, npt.NDArray[np.float64] | Tensor] = {}

    def __init__(self,
                 dataset: SpatialDataset,
                 logger: ILogger,
                 optimizeMode: Literal['cuda', 'cpu'] = 'cuda',
                 kernel_type: KernelFunctionType = 'bisquare',
                 kernel_bandwidth_type: KernelBandwidthType = 'adaptive'
                 ) -> None:
        """
        Initializes the Kernel with a dataset, bandwidth, and kernel type.

        Args:
            dataset (SpatialDataset): The spatial dataset for generating the weighted matrix.
            bandwidth (float): The bandwidth parameter for controlling spatial influence.
            kernel_type (KernelFunctionType, optional): The kernel function type to use; 
                defaults to 'triangular'.
        """
        self.dataset = dataset
        self.logger = logger
        self.optimizeMode = optimizeMode
        self.kernel_type = kernel_type
        self.kernel_bandwidth_type = kernel_bandwidth_type

        self.__init_distance_vectors()
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

    def get_weighted_matrix_by_id(self, index: int) -> npt.NDArray[np.float64] | Tensor:
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

    def get_distance_vector_by_id(self, index: int) -> npt.NDArray[np.float64] | Tensor:
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
        if isinstance(self.bandwidth, Tensor):
            distance_vector = self.__calculate_distance_vector_torch(index)
            if isinstance(distance_vector, Tensor):
                self.__calculate_weighted_matrix_torch(
                    index,
                    distance_vector
                )

        if isinstance(self.bandwidth, float):
            distance_vector = self.__calculate_distance_vector(index)
            if not isinstance(distance_vector, Tensor):
                self.__calculate_weighted_matrix(
                    index,
                    distance_vector
                )

    def __init_distance_vectors(self):
        if self.dataset is None:
            raise ValueError("Dataset is not setup in Kernel")
        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not setup in Kernel")

        if self.optimizeMode == 'cuda':
            for index in range(0, len(self.dataset.dataPoints)):
                self.__calculate_distance_vector_torch(index)
        else:
            for index in range(0, len(self.dataset.dataPoints)):
                self.__calculate_distance_vector(index)

    def __calculate_distance_vector(self, index: int) -> npt.NDArray[np.float64] | Tensor:
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

    def __calculate_distance_vector_torch(self, index: int) -> npt.NDArray[np.float64] | Tensor:
        """
        Retrieves the distance vector for a specific data point index.

        This function calculates the distances from a specified data point to all other points 
        in the dataset using PyTorch tensors.

        Args:
            index (int): The index of the data point to calculate distances from.

        Returns:
            torch.Tensor: A 1D tensor of distances from the specified data point to all other points.

        Raises:
            ValueError: If the dataset or data points are not initialized.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not setup in Kernel")
        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not setup in Kernel")

        # Retrieve the distance vector from cache if it exists
        if index in self.distance_vector_cache:
            return self.distance_vector_cache[index]

        # Calculate the distance vector and store in cache
        # distance_vector_i = torch.tensor(
        #     get_2d_distance_vector_torch(index, self.dataset),
        #     dtype=torch.float32
        # ).to('cuda')

        distance_vector_i = get_2d_distance_vector_torch(index, self.dataset)

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

    def __calculate_weighted_matrix_torch(self,
                                          index: int,
                                          distance_vector: Tensor,
                                          eps=1.0000001):
        """
        Calculates the weighted matrix based on the kernel function and distance vector."
        """
        if self.bandwidth is None:
            raise ValueError("Bandwidth is not set up in Kernel")
        if not isinstance(self.bandwidth, Tensor):
            raise ValueError(
                "__calculate_weighted_matrix_torch is only for gpu, bandwidth needs to be a tensor")

        distance_bandwidth = self.bandwidth
        if self.kernel_bandwidth_type == 'adaptive':
            distance_bandwidth = torch.kthvalue(
                distance_vector, int(self.bandwidth))[0] * eps

        zs: Tensor = distance_vector / distance_bandwidth
        weighted_matrix_i = self.__spatial_weights(
            zs, distance_vector, distance_bandwidth)

        # store the weighted matrix in the cache
        self.weighted_matrix_cache[index] = weighted_matrix_i.reshape(-1, 1)

    def __spatial_weights(self,
                          zs: Tensor,
                          distance_vector: Tensor,
                          distance_bandwidth: Tensor
                          ):
        weighted_matrix_i = torch.zeros_like(
            zs, dtype=torch.float32, requires_grad=True).to('cuda')

        if self.kernel_type == 'triangular':
            weighted_matrix_i = 1 - zs
        elif self.kernel_type == 'uniform':
            weighted_matrix_i = torch.ones(
                zs.shape, dtype=torch.float32, requires_grad=True).to('cuda') * 0.5
        elif self.kernel_type == 'quadratic':
            weighted_matrix_i = (3. / 4) * (1 - zs**2)
        elif self.kernel_type == 'quartic':
            weighted_matrix_i = (15. / 16) * (1 - zs**2)**2
        elif self.kernel_type == 'gaussian':
            weighted_matrix_i = torch.exp(-0.5 * (zs)**2)
        elif self.kernel_type == 'bisquare':
            weighted_matrix_i = (1 - (zs)**2)**2
        elif self.kernel_type == 'exponential':
            weighted_matrix_i = torch.exp(-zs)
        else:
            raise ValueError('Unsupported kernel function')

        if self.kernel_type == 'bisquare':
            weighted_matrix_i[(distance_vector >= distance_bandwidth)] = 0

        return weighted_matrix_i
