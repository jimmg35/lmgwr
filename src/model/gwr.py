import numpy as np
import numpy.typing as npt
from scipy import linalg
from tqdm import tqdm

from src.model.base import Base
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.gwr_logger import GwrLogger
from src.kernel.ikernel import IKernel
from src.log.ilogger import ILogger


class GWR(Base):
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
                 kernel: GwrKernel | LgwrKernel) -> None:
        super().__init__(dataset, kernel)
        self.model_type = "GWR"

    def fit(self):
        """ 
        Fit the GWR model with the provided dataset and spatial weights based on the kernel.
        This method iterates over each data point in the dataset and calculates local regression
        coefficients using spatial weights, implementing the core concept of GWR.
        """

        super()._init_estimates()

        for index in range(len(self.dataset)):
            self._local_fit(index)

        
        super()._calculate_residuals()
        super()._calculate_mu()
        super()._calculate_llf()
        super()._calculate_tr_S()
        super()._calculate_r_squared()
        super()._calculate_aic_aicc()
        print("GWR : GWR model fitting is complete.")

        return self

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
