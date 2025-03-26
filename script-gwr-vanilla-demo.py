import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    dataframe = pd.read_csv(r'./data/income_health.csv')
    spatialDataset = SpatialDataset(
        dataframe,
        IFieldInfo(
            predictor_fields=['pct_bach', 'pct_black', 'median_income'],
            response_field='DIABETES',
            coordinate_x_field='proj_X',
            coordinate_y_field='proj_Y'
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
    # Or you could call the update function in gwr (this interface is mainly built for optimizer)
    # gwr.update_bandwidth(117).fit()

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()

    logger.save_model_info_json()
