
import numpy as np
import numpy.typing as npt
import numpy.linalg as la
from scipy import sparse as sp
from spreg.utils import spdot, spmultiply
from src.dataset.spatial_dataset import SpatialDataset
# from src.estimates


class OLS():

    dataset: SpatialDataset
    estimates:

    def __init__(self, dataset: SpatialDataset) -> None:
        self.dataset = dataset

    def fit(self):
        beta = self.__compute_betas(self.dataset.y, self.dataset.x_matrix)
        y_hat = np.dot(self.dataset.x_matrix, beta)
        residuels = self.dataset.y - y_hat

    def __compute_betas(self, y, x):
        xT = x.T
        xtx = spdot(xT, x)
        xtx_inv = la.inv(xtx)
        xtx_inv = sp.csr_matrix(xtx_inv)
        xTy = spdot(xT, y, array_out=False)
        beta = spdot(xtx_inv, xTy)
        return beta
