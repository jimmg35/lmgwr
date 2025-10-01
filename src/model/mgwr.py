
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

        # P = []
        # Q = []
        # I = np.eye(self.dataset.n)
        b = []
        for j_1 in range(self.dataset.k):
            self.dataset.use_column(j_1)
            gwr = GWR(self.dataset, self.kernel).update_bandwidth(
                self.bandwidth_set[j_1]).fit()
            b.append(gwr.betas)

        comB = np.hstack(b)

        print(comB)
        # Pj = []

    def update_bandwidth_set(self, bandwidth_set):
        self.bandwidth_set = bandwidth_set
        return self
