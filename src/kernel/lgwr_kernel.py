from torch import Tensor

from src.kernel.gwr_kernel import GwrKernel
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import KernelFunctionType, KernelBandwidthType
from src.log.lgwr_logger import LgwrLogger
from src.optimizer.lgwr_optimizer import LgwrOptimizeMode


class LgwrKernel(GwrKernel):

    optimizeMode: LgwrOptimizeMode

    def __init__(self,
                 dataset: SpatialDataset,
                 logger: LgwrLogger,
                 optimizeMode: LgwrOptimizeMode = 'cuda',
                 kernel_type: KernelFunctionType = 'bisquare',
                 kernel_bandwidth_type: KernelBandwidthType = 'adaptive'
                 ):
        super().__init__(
            dataset,
            logger,
            kernel_type,
            kernel_bandwidth_type
        )
        self.optimizeMode = optimizeMode

    def update_local_bandwidth(self, index: int, bandwidth: float | Tensor):
        """
        Update the bandwidth value for the LGWR model.

        This method updates the bandwidth value used by the kernel to calculate spatial weights.
        It is typically used when optimizing the bandwidth for the GWR model.

        Args:
            bandwidth (float): The new bandwidth value to use for the GWR model.

        Raises:
            ValueError: If the kernel is not set up in the GWR model.
        """

        self.bandwidth = bandwidth
        super().update_weighted_matrix_by_id(index)
