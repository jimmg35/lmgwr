
import numpy as np

from src.model.base import Base
from src.model.gwr import GWR
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.ikernel import IKernel
from src.kernel.gwr_kernel import GwrKernel


class MGWR(Base):

    bandwidth_set: list[float]

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: GwrKernel) -> None:
        super().__init__(dataset, kernel)
        self.bandwidth_set = []

    def exact_fit(self) -> None:
        """ 
        Fit the MGWR model with the provided dataset and spatial weights based on the kernel.
        This method iterates over each data point in the dataset and calculates local regression
        coefficients using spatial weights, implementing the core concept of MGWR.
        """

        if not isinstance(self.kernel, GwrKernel):
            raise TypeError("kernel must be of type GwrKernel")

        P = []
        Q = []
        I = np.eye(self.dataset.n)
        self.kernel
        # for j_1 in range(self.dataset.n):

        #     Aj = GWR(self.dataset, self.kernel).update_bandwidth()
        #     Pj = []

    def update_bandwidth_set(self, bandwidth_set):
        self.bandwidth_set = bandwidth_set
