import numpy as np
import pandas as pd

from src.log.logger import create_logger
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_bandwidth_optimizer import GwrBandwidthOptimizer

create_logger()


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        isSpherical=False
    )

    kernel = GwrKernel(spatialDataset, 'bisquare')
    gwr = GWR(spatialDataset, kernel)

    kernel.update_bandwidth(117)
    gwr.fit()

    # optimizer = GwrBandwidthOptimizer(gwr, kernel)
    # optimal_bandwidth = optimizer.optimize()
    # print(f'Optimal bandwidth: {optimal_bandwidth}')
