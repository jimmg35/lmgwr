import numpy as np
import numpy.typing as npt
from scipy import linalg
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from tqdm import tqdm
import logging


import numpy.linalg as la
from scipy import sparse as sp
from spreg.utils import spdot, spmultiply


class LGWR:
    dataset: SpatialDataset
    kernel: GwrKernel

    def __init__(self, dataset: SpatialDataset, kernel: GwrKernel) -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """
        self.dataset = dataset
        self.kernel = kernel
