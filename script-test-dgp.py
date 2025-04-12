import statsmodels.api as sm
from src.dgp.simulation import generate_processes, generate_data
from src.dgp.plot import plot_s
import numpy as np
from src.model.gwr import GWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset

if __name__ == "__main__":

    simulated_spatial_dataset = SimulatedSpatialDataset()


# size = 40

# # generate real processes
# b0, b1, b2 = generate_processes(size=size)

# # generate data
# X1, X2, coordinates = generate_data(size=size)

# # generate the ground truth
# err = np.random.randn(size * size)
# y = (b0 + b1 * X1 + b2 * X2 + err).reshape(-1, 1)


# # Prepare the design matrix with an intercept
# X = np.hstack([np.ones((size * size, 1)),
#                X1.reshape(-1, 1), X2.reshape(-1, 1)])

# # Fit the OLS model
# ols_model = sm.OLS(y, X).fit()

# # Print the summary of the model
# print(ols_model.summary())
# print(ols_model.params)


# plot_s(
#     np.vstack([b0, b1, b2]),
#     [r"True $\beta_0$", r"True $\beta_1$", r"True $\beta_2$"],
#     vmin=0,
#     vmax=4,
#     size=size
# )
