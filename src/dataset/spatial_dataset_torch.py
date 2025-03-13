from pandas import DataFrame

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.dataset.spatial_dataset import IFieldInfo
# from src.utility.optimize_mode import OptimizeMode


class SpatialDataset(Dataset):

    X: Tensor
    y: Tensor
    optimizeMode: torch.device

    def __init__(self,
                 dataframe: DataFrame,
                 fieldInfo: IFieldInfo,
                 optimizeMode: torch.device
                 ):

        self.optimizeMode = optimizeMode

        self.X = torch.tensor(
            dataframe[fieldInfo.predictor_fields].values,
            dtype=torch.float32
        ).to(self.optimizeMode)

        self.y = torch.tensor(
            dataframe[fieldInfo.response_field].values,
            dtype=torch.float32
        ).unsqueeze(1).to(self.optimizeMode)

        coordinates = torch.tensor(
            dataframe[[fieldInfo.coordinate_x_field,
                       fieldInfo.coordinate_y_field]].values,
            dtype=torch.float32
        ).to(self.optimizeMode)

        distance_matrix = torch.cdist(
            coordinates, coordinates, p=2
        ).to(self.optimizeMode)

        self.distance_matrix = distance_matrix  # (n, n)
        # (n, 1)
        # (n, k)
        self.n = distance_matrix.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.distance_matrix[index], self.y[index]
