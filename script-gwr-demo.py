import numpy as np
import pandas as pd

from src.log.logger import create_logger
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_bandwidth_optimizer import GwrBandwidthOptimizer
from src.log.logger import GwrLogger


if __name__ == '__main__':

    logger = GwrLogger()

    georgia_data = pd.read_csv(r'./data/GData_utm.csv')

    spatialDataset = SpatialDataset(
        georgia_data,
        IFieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        logger,
        isSpherical=False
    )

    kernel = GwrKernel(spatialDataset, logger, 'bisquare')
    gwr = GWR(spatialDataset, logger, kernel)

    # Manually update bandwidth and fit the model.
    # kernel.update_bandwidth(117)
    # gwr.fit()

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrBandwidthOptimizer(gwr, kernel)
    optimal_bandwidth = optimizer.optimize()
