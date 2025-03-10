
import numpy as np
import numpy.typing as npt
from typing import Literal, TypeAlias, Dict

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.distance.get_2d_distance_vector import get_2d_distance_vector
from src.log.gwr_logger import GwrLogger

KernelFunctionType: TypeAlias = Literal['triangular', 'uniform', 'quadratic',
                                        'quartic', 'gaussian', 'bisquare', 'exponential']
KernelBandwidthType: TypeAlias = Literal['distance_based', 'adaptive']


# This is an interface class for kernel
class IKernel:

    logger: GwrLogger
    dataset: SpatialDataset | None = None
    bandwidth: float | None = None
    kernel_type: KernelFunctionType = "bisquare"
    kernel_bandwidth_type: KernelBandwidthType = "adaptive"
    weighted_matrix_cache: Dict[int, npt.NDArray[np.float64]] = {}
    distance_vector_cache: Dict[int, npt.NDArray[np.float64]] = {}

    def update_bandwidth(self, bandwidth: float) -> None:
        raise NotImplementedError("Method not implemented")

    def update_local_bandwidth(self, index: int, bandwidth: float):
        raise NotImplementedError("Method not implemented")

    def get_weighted_matrix_by_id(self, index: int) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Method not implemented")

    def get_distance_vector_by_id(self, index: int) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Method not implemented")

    def update_weighted_matrix_by_id(self, index: int) -> None:
        raise NotImplementedError("Method not implemented")
