import numpy as np
import numpy.typing as npt

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.ikernel import IKernel
# from src.log.


class IModel:
    dataset: SpatialDataset
    kernel: IKernel
    logger: GwrLogger

    # estimates for each data point
    betas: npt.NDArray[np.float64]
    y_hats: npt.NDArray[np.float64]
    S: npt.NDArray[np.float64]  # hat matrix
    residuals: npt.NDArray[np.float64]

    # matrices for the GWR model
    r_squared: float
    aic: float
    aicc: float
