import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.idataset import FieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.utility.distribution_test import normal_distribution_test


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    field_size = 40
    dataset = SimulatedSpatialDataset(field_size=field_size)
    [X] = dataset.generate_data()
    [beta] = dataset.generate_processes()
    [y, err] = dataset.fit_y(X, beta)

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(dataset, 'bisquare')
    gwr = GWR(dataset, kernel, logger)

    gwr.update_bandwidth(100).fit()

    # # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    # optimizer = GwrOptimizer(gwr, kernel, logger)
    # optimal_bandwidth = optimizer.optimize()

    # dataset.plot(
    #     b=beta.T,
    #     sub_title=['b0', 'b1', 'b2'],
    #     size=field_size
    # )
    # dataset.plot(
    #     b=gwr.betas.T,
    #     sub_title=['b0_gwr', 'b1_gwr', 'b2_gwr'],
    #     size=field_size
    # )

    # # pred_y = (b0_gwr * dataset.X[:, 0] + b1_gwr * dataset.X[:, 1] + b2_gwr *
    # #             dataset.X[:, 2] + dataset.err).reshape(-1, 1)
    # # pred_err = pred_y - y
    # # # dataset.plot(
    # # #     b=np.vstack(pred_err.T),
    # # #     sub_title=['prediction error'],
    # # #     size=field_size
    # # # )
    # # normal_distribution_test(pred_err, title='Prediction Error', xlabel='Error')
