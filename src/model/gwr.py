import numpy as np
import numpy.typing as npt
from scipy import linalg
from tqdm import tqdm

from src.model.imodel import IModel
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.gwr_logger import GwrLogger
from src.kernel.ikernel import IKernel
from src.log.ilogger import ILogger


class GWR(IModel):
    """
    A class for performing Geographically Weighted Regression (GWR) using a spatial dataset.

    This class manages the process of fitting a GWR model, which allows for spatially varying
    relationships between predictors and the response variable. It relies on a spatial dataset
    and a kernel for generating location-based weights.

    Attributes:
        dataset (SpatialDataset): The spatial dataset used for the GWR analysis.
        kernel (GwrKernel): The kernel function that defines spatial weights for each location.
    """

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: IKernel,
                 logger: ILogger) -> None:
        super().__init__(dataset, kernel, logger)

    def fit(self) -> None:
        """ 
        Fit the GWR model with the provided dataset and spatial weights based on the kernel.
        This method iterates over each data point in the dataset and calculates local regression
        coefficients using spatial weights, implementing the core concept of GWR.
        """

        super()._init_estimates()

        for index in range(len(self.dataset)):
            self._local_fit(index)

        # update estimates (outside of loop for calculations)
        self.residuals = self.dataset.y - self.y_hats.reshape(-1, 1)

        super()._calculate_r_squared()
        super()._calculate_aic_aicc()

    def update_bandwidth(self, bandwidth: float):
        """
        Update the bandwidth value for the GWR model.

        This method updates the bandwidth value used by the kernel to calculate spatial weights.
        It is typically used when optimizing the bandwidth for the GWR model.

        Args:
            bandwidth (float): The new bandwidth value to use for the GWR model.

        Raises:
            ValueError: If the kernel is not set up in the GWR model.
        """
        if self.kernel is None:
            raise ValueError("Kernel is not set up in the GWR model")

        self.kernel.update_bandwidth(bandwidth)
        return self
