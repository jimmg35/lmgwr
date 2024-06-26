import os
import logging
import pandas as pd

from src.dataset.impl import SpatialDataset
from src.dataset.interface import ISpatialDatasetParams, IFieldInfo


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debkkug.log"),  # 日志输出到文件
                        logging.StreamHandler()           # 日志输出到控制台
                    ])
logger = logging.getLogger(__name__)


current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./src/dataset/data/synthetic_dataset.csv')

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
