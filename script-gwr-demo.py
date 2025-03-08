import numpy as np
import pandas as pd

from src.log.logger import create_logger
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel

create_logger()


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],
            response_field='PctPov',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        isSpherical=True
    )

    kernel = GwrKernel(spatialDataset, 'bisquare')
    gwr = GWR(spatialDataset, kernel)

    kernel.update_bandwidth(100)
    gwr.fit()
    kernel.update_bandwidth(200)
    gwr.fit()
    kernel.update_bandwidth(300)
    gwr.fit()
    kernel.update_bandwidth(400)
    gwr.fit()
    kernel.update_bandwidth(500)
    gwr.fit()
