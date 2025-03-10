import numpy as np
import numpy.typing as npt

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.lgwr_logger import LgwrLogger
from src.model.gwr import GWR


class LGWR(GWR):

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: LgwrKernel,
                 logger: LgwrLogger) -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """

        super().__init__(dataset, logger, kernel)

    def update_local_bandwidth(self, index: int, bandwidth: float):
        self.kernel.update_local_bandwidth(index, bandwidth)
        return self
