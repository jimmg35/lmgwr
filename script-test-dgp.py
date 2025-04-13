import statsmodels.api as sm
from src.dgp.simulation import generate_processes, generate_data
from src.dgp.plot import plot_s
import numpy as np
from src.model.gwr import GWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger

if __name__ == "__main__":

    logger = GwrLogger()
    field_size = 40
    err = np.random.randn(field_size * field_size)

    dataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = dataset.generate_processes()
    [X, y] = dataset.fit_y(b0, b1, b2)

    dataset.plot(
        b=np.vstack([b0, b1, b2]),
        sub_title=['b0', 'b1', 'b2'],
        size=field_size
    )
    # dataset.plot(
    #     b=np.vstack([X.T, y.T]),
    #     sub_title=['intercept', 'X1', "X2", 'y'],
    #     size=field_size
    # )

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(dataset, 'bisquare')
    gwr = GWR(dataset, kernel, logger)

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()

    [b0_gwr, b1_gwr, b2_gwr] = [gwr.betas[:, 0], gwr.betas[:, 1], gwr.betas[:, 2]]

    dataset.plot(
        b=np.vstack([b0_gwr, b1_gwr, b2_gwr]),
        sub_title=['b0_gwr', 'b1_gwr', 'b2_gwr'],
        size=field_size
    )

    # logger.save_model_info_json()


# size = 40


# # Prepare the design matrix with an intercept
# X = np.hstack([np.ones((size * size, 1)),
#                X1.reshape(-1, 1), X2.reshape(-1, 1)])

# # Fit the OLS model
# ols_model = sm.OLS(y, X).fit()

# # Print the summary of the model
# print(ols_model.summary())
# print(ols_model.params)
