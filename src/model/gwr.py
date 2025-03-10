import numpy as np
import numpy.typing as npt
from scipy import linalg
from tqdm import tqdm

from src.model.imodel import IModel
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.gwr_logger import GwrLogger


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
    # dataset: SpatialDataset
    # kernel: GwrKernel
    # logger: GwrLogger

    # # estimates for each data point
    # betas: npt.NDArray[np.float64]
    # y_hats: npt.NDArray[np.float64]
    # S: npt.NDArray[np.float64]  # hat matrix
    # residuals: npt.NDArray[np.float64]

    # # matrices for the GWR model
    # r_squared: float
    # aic: float
    # aicc: float

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: GwrKernel,
                 logger: GwrLogger) -> None:
        super().__init__(dataset, kernel, logger)

    def fit(self) -> None:
        """ 
        # 這是備忘錄
        lgwr每次fit要傳入該點帶寬以及該點的index
        gwr這邊的fit是一次性fit全部的點
        而lgwr是一次fit一個點 (不一定採納)

        傾向將帶寬寫成一個class (否決 記憶體占用太大)
        在optimizer中 若是gwr則指傳入一個 帶寬
        若是lgwr則傳入n個帶寬class {index: 帶寬}
        lmgwr {index: [帶寬1, 帶寬2, ...]}
        [帶寬1, 帶寬2, ...] lmsb


        Fit the GWR model with the provided dataset and spatial weights based on the kernel.
        This method iterates over each data point in the dataset and calculates local regression
        coefficients using spatial weights, implementing the core concept of GWR.

        Raises:
            ValueError: If `dataPoints` are not set up in the dataset.
            NotImplementedError: If the method's fitting logic is not fully implemented.
        """
        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not set up in the dataset")

        super()._init_estimates()

        for index in range(len(self.dataset.dataPoints)):
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
