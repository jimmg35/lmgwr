import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_bandwidth_optimizer import GwrBandwidthOptimizer
from src.log.gwr_logger import GwrLogger


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
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

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(spatialDataset, logger, 'bisquare')
    gwr = GWR(spatialDataset, kernel, logger)

    # Manually update bandwidth and fit the model.
    # kernel.update_bandwidth(117)
    # gwr.fit()
    # Or you could call the update function in gwr
    # gwr.update_bandwidth(117).fit()
    # this interface is mainly built for optimizer

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrBandwidthOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()

    logger.save_model_info_json()
