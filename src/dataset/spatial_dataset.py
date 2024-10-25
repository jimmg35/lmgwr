import pandas as pd
from .interfaces.interface import ISpatialDatasetParams, IFieldInfo
from .dataset import Dataset


class SpatialDataset(Dataset):

    def __init__(self, param: ISpatialDatasetParams):
        super().__init__(param)


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
