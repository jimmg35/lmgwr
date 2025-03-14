import pandas as pd

import torch
from torch.nn import MSELoss, L1Loss

from src.dataset.spatial_dataset import IFieldInfo
from src.dataset.spatial_dataset_torch import SpatialDataset as SpatialDatasetTorch
from src.model.lgwr_torch import LGWR
from src.log.lgwr_logger import LgwrLogger
from src.optimizer.lgwr_optimizer_torch import LgwrOptimizer


if __name__ == "__main__":

    optimizeMode = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Data
    georgia_dataframe = pd.read_csv(r'./data/GData_utm.csv')
    dataset_torch = SpatialDatasetTorch(
        georgia_dataframe,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],
            response_field='PctPov',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        optimizeMode
    )

    # Initialize components and hyperparameters
    model = LGWR(dataset_torch)
    logger = LgwrLogger()

    loss_function = MSELoss()
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 1

    # Prepare the training container
    optimizer = LgwrOptimizer(
        model,
        logger,
        dataset_torch,
        loss_function,
        optimizeMode,
        LEARNING_RATE,
        EPOCHS,
        BATCH_SIZE
    )

    optimizer.train()
