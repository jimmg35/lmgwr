import pandas as pd

from src.log.logger import create_logger
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel

create_logger()


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/synthetic_dataset.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['temperature', 'moisture'],
            response_field='pm25',
            coordinate_x_field='coor_x',
            coordinate_y_field='coor_y'
        ),
        isSpherical=True
    )

    

    gwr = GWR(spatialDataset)
