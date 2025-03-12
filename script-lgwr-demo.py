import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.log.lgwr_logger import LgwrLogger
from src.model.lgwr import LGWR
from src.kernel.lgwr_kernel import LgwrKernel
from src.optimizer.lgwr_optimizer import LgwrOptimizer, LBNN


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

    kernel = LgwrKernel(spatialDataset, logger, kernel_type='bisquare')
    lgwr = LGWR(spatialDataset, kernel, logger)
    lbnn = LBNN(spatialDataset)

    optimizer = LgwrOptimizer(
        lgwr, lbnn,
        kernel,
        logger,
        lr=0.01,
        epochs=1
    )

    optimizer.optimize()

    # you could update the bandwidth and weighted matrix for a singel location
    # the fit function will still calculate the estimates of all locations
    # lgwr.update_local_bandwidth(0, 117).fit()

    # or you could update the bandwidth for all locations
    # and call the fit function for once
    # to reduce the computational demands
    # lgwr.update_local_bandwidth(0, 117)
    # lgwr.update_local_bandwidth(1, 117)
    # lgwr.update_local_bandwidth(2, 117)
    # lgwr.fit()

    # kernel.update_local_bandwidth(0, 117)
    # gwr.fit()

    logger.save_model_info_json()
