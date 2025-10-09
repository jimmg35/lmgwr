
import numpy as np

from src.model.base import Base
from src.model.gwr import GWR
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.ikernel import IKernel
from src.kernel.gwr_kernel import GwrKernel


class MGWR(Base):

    bandwidth_set: list[float] | None

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: GwrKernel) -> None:
        super().__init__(dataset, kernel)
        self.bandwidth_set = None

    def exact_fit(self) -> None:
        """ 
        Fit the MGWR model with the provided dataset and spatial weights based on the kernel.
        This method iterates over each data point in the dataset and calculates local regression
        coefficients using spatial weights, implementing the core concept of MGWR.
        """

        if not isinstance(self.kernel, GwrKernel):
            raise TypeError("kernel must be of type GwrKernel")
        if self.bandwidth_set is None:
            raise ValueError(
                "bandwidth_set must be provided before fitting the model")

        P = []
        Q = []
        I = np.eye(self.dataset.n)
        for j_1 in range(self.dataset.k):
            self.dataset.use_column(j_1)
            Aj = GWR(self.dataset, self.kernel).update_bandwidth(
                self.bandwidth_set[j_1]).fit().S
            Pj = []
            for j_2 in range(self.dataset.k):
                if j_1 == j_2:
                    Pj.append(I)
                else:
                    Pj.append(Aj)
            P.append(Pj)
            Q.append([Aj])

        P = np.block(P)
        Q = np.block(Q)
        R = np.linalg.solve(P, Q)
        f = R.dot(self.dataset.y)

        self.dataset.reset_columns()
        params = f / self.dataset.X.T.reshape(-1, 1)
        params = params.reshape(-1, self.dataset.n).T

        print(params)

        R = np.stack(np.split(R, self.dataset.k), axis=2)
        ENP_j = np.trace(R, axis1=0, axis2=1)
        predy = np.sum(self.dataset.X * params, axis=1).reshape(-1, 1)
        w = np.ones(self.dataset.n)

        CCT = np.zeros((self.dataset.n, self.dataset.k))
        for j in range(self.dataset.k):
            CCT[:, j] = (
                (R[:, :, j] / self.dataset.X[:, j].reshape(-1, 1))**2).sum(axis=1)

    def update_bandwidth_set(self, bandwidth_set):
        self.bandwidth_set = bandwidth_set
        return self
