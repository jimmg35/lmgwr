import numpy as np

from src.dataset.spatial_dataset import SpatialDataset
from src.model.gwr import GWR
from src.kernel.ikernel import IKernel
from src.log.ilogger import ILogger


class LGWR(GWR):

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: IKernel,
                 logger: ILogger) -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """

        super().__init__(dataset, kernel, logger)

        self.y_hats = np.zeros(
            self.dataset.x_matrix.shape[0], dtype=np.float64
        )
        self.residuals = np.zeros(
            self.dataset.x_matrix.shape[0], dtype=np.float64
        )

    def update_local_bandwidth(self, index: int, bandwidth: float):
        self.kernel.update_local_bandwidth(index, bandwidth)
        return self
