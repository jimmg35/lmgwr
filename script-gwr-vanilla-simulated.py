import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.idataset import FieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    field_size = 40
    spatialDataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = spatialDataset.generate_processes()
    [X, y] = spatialDataset.fit_y(b0, b1, b2)

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(spatialDataset, 'bisquare')
    gwr = GWR(spatialDataset, kernel, logger)

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()
