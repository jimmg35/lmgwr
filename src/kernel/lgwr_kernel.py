from src.kernel.gwr_kernel import GwrKernel
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import KernelFunctionType, KernelBandwidthType
from src.log.lgwr_logger import LgwrLogger
from src.utility.overrides import overrides


class LgwrKernel(GwrKernel):
    def __init__(self,
                 dataset: SpatialDataset,
                 logger: LgwrLogger,
                 kernel_type: KernelFunctionType = 'bisquare',
                 kernel_bandwidth_type: KernelBandwidthType = 'adaptive'
                 ):
        super().__init__(
            dataset,
            logger,
            kernel_type,
            kernel_bandwidth_type
        )

    def update_local_bandwidth(self, index: int, bandwidth: float):
        # 要有一個temp_bandwidth變數用於該location
        self.bandwidth = bandwidth

        super().update_weighted_matrix_by_id(index)
        # print(super().bandwidth)

    def __calculate_weighted_matrix(self):
        print("override")
