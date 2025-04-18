
# acknowledgements to Taylor Oshan for part of the source code

import numpy as np
import pandas as pd
import numpy.typing as npt

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import FieldInfo
from src.kernel.ikernel import IKernel, KernelFunctionType, KernelBandwidthType


class GwrKernel(IKernel):
    """
    GWR kernel function specifications.

    This class builds a spatial weighted matrix for each data point, using a specified
    kernel function to calculate weights based on the distance between data points.

    Attributes:
        dataset (SpatialDataset | None): The spatial dataset used to calculate weights.
        bandwidth (float): The bandwidth parameter controlling the kernel's spatial influence.
        kernel_type (KernelFunctionType): The type of kernel function to use for weight calculations.
    """

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel_type: KernelFunctionType = 'bisquare',
                 kernel_bandwidth_type: KernelBandwidthType = 'adaptive'
                 ) -> None:
        super().__init__(dataset, 'cpu', kernel_type, kernel_bandwidth_type)

    def update_bandwidth(self, bandwidth: float) -> None:
        """
        Updates the bandwidth parameter for the kernel.
        this function will update the bandwidth and weighted matrix of 
        all locations

        """
        if self.dataset is None:
            raise ValueError(
                "GwrKernel: Dataset is not setup in Kernel, Couldn't update the bandwidth")

        self.bandwidth = bandwidth

        for i in range(len(self.dataset)):
            self.update_weighted_matrix_by_id(i)


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/synthetic_dataset.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        FieldInfo(
            predictor_fields=['temperature', 'moisture'],
            response_field='pm25',
            coordinate_x_field='coor_x',
            coordinate_y_field='coor_y'
        ),
        isSpherical=True
    )

    gwrKernel = GwrKernel(spatialDataset, 'triangular')
    gwrKernel.update_bandwidth(100)
    wi = gwrKernel.get_weighted_matrix_by_id(0)
