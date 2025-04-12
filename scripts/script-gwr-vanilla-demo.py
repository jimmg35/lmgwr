import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.idataset import FieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    georgia_data = pd.read_csv(r'./data/GData_utm.csv')
    spatialDataset = SpatialDataset(
        georgia_data,
        FieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        logger=logger,
        isSpherical=False
    )

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(spatialDataset, 'bisquare')
    gwr = GWR(spatialDataset, kernel, logger)

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()

    logger.save_model_info_json()
