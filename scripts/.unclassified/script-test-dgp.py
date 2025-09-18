
import numpy as np
from src.model.gwr import GWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger

if __name__ == "__main__":


    ## Create a simulated dataset.
    field_size = 40
    dataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = dataset.generate_processes()
    [X, y, err] = dataset.fit_y(b0, b1, b2)
    # dataset.plot(
    #     b=np.vstack([b0, b1, b2]),
    #     sub_title=['b0', 'b1', 'b2'],
    #     size=field_size
    # )

    ## Create a GWR kernel and GWR model.
    logger = GwrLogger()
    kernel = GwrKernel(dataset, 'bisquare')
    gwr = GWR(dataset, kernel, logger)

    ## Use the vanilla bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()

    [b0_gwr, b1_gwr, b2_gwr] = [gwr.betas[:, 0], gwr.betas[:, 1], gwr.betas[:, 2]]

    print(np.array([b0_gwr, b1_gwr, b2_gwr]).shape)
    dataset.plot(
        b=np.vstack([b0_gwr, b1_gwr, b2_gwr]),
        sub_title=['b0_gwr', 'b1_gwr', 'b2_gwr'],
        size=field_size
    )
