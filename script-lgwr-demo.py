import numpy as np
import pandas as pd

from src.log.lgwr_logger import LgwrLogger
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.kernel.lgwr_kernel import LgwrKernel


if __name__ == '__main__':

    logger = LgwrLogger()

    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')
    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],
            response_field='PctPov',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        logger,
        isSpherical=True
    )

    kernel = LgwrKernel(spatialDataset, logger, 'bisquare')

    kernel.update_local_bandwidth(0, 117)
    # gwr = GWR(spatialDataset, kernel)
    # gwr.fit()

    logger.save_model_info_json()
