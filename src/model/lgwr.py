import numpy as np

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.ilogger import ILogger
from src.model.gwr import GWR


class LGWR(GWR):

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: LgwrKernel,
                 logger: ILogger) -> None:
        """
        Initialize the LGWR model with the provided dataset, kernel, and logger.

        Args:
            dataset (SpatialDataset): The spatial dataset used for the LGWR analysis.
            kernel (LgwrKernel): The kernel function that defines spatial weights for each location.
            logger (ILogger): The logger object used to record information about the LGWR model.

        Raises:
            ValueError: If the dataset or kernel is not set up in the LGWR model.
        """

        super().__init__(dataset, kernel, logger)

        self.y_hats = np.zeros(
            self.dataset.X.shape[0], dtype=np.float64
        )
        self.residuals = np.zeros(
            self.dataset.X.shape[0], dtype=np.float64
        )

    def update_local_bandwidth(self, index: int, bandwidth: float):
        self.kernel.update_local_bandwidth(index, bandwidth)
        return self

    def update_local_bandwidth_vector(self, bandwidth_vector: np.ndarray):
        for index in range(0, self.dataset.X.shape[0]):
            self.kernel.update_local_bandwidth(index, bandwidth_vector[index])
        return self
