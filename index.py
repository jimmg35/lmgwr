import os
import logging
import pandas as pd

from src.dataset.impl import SpatialDataset
from src.dataset.interface import ISpatialDatasetParams, IFieldInfo
from src.log.timestamp import current_time_str

log_path = os.path.join(os.getcwd(), 'logs',  f'{current_time_str()}-log')
if os.path.exists(log_path) is False:
    os.makedirs(log_path)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(
                            os.path.join(log_path, 'record.log')),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/synthetic_dataset.csv')

    spatialDataset = SpatialDataset(
        ISpatialDatasetParams(
            fieldInfo=IFieldInfo(
                predictor_fields=['temperature', 'moisture'],
                response_field='pm25',
                coordinate_x_field='coor_x',
                coordinate_y_field='coor_y'
            ), data=synthetic_data
        )
    )
