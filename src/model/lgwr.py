import numpy as np
import numpy.typing as npt

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.logger import GwrLogger
from src.model.gwr import GWR


class LGWR(GWR):

    dataset: SpatialDataset
    kernel: GwrKernel
    logger: GwrLogger

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: GwrKernel,
                 logger: GwrLogger) -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """

        super().__init__(dataset, logger, kernel)
