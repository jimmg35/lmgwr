import numpy as np
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.idataset import FieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from scipy import stats


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    field_size = 40
    dataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = dataset.generate_processes()
    [X, y, err] = dataset.fit_y(b0, b1, b2)

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(dataset, 'bisquare')
    gwr = GWR(dataset, kernel, logger)

    # Use the bandwidth optimizer to automatically find the optimal bandwidth.
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()

    [b0_gwr, b1_gwr, b2_gwr] = [gwr.betas[:, 0], gwr.betas[:, 1], gwr.betas[:, 2]]
    # dataset.plot(
    #     b=np.vstack([b0_gwr, b1_gwr, b2_gwr]),
    #     sub_title=['b0_gwr', 'b1_gwr', 'b2_gwr'],
    #     size=field_size
    # )

    
    pred_y = (b0_gwr * dataset.X[:, 0] + b1_gwr * dataset.X[:, 1] + b2_gwr *
                dataset.X[:, 2] + dataset.err).reshape(-1, 1)
    pred_err = pred_y - y
    # dataset.plot(
    #     b=np.vstack(pred_err.T),
    #     sub_title=['prediction error'],
    #     size=field_size
    # )

    import matplotlib.pyplot as plt

    # Plot histogram of pred_err
    plt.hist(pred_err, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of Prediction Error')
    plt.xlabel('Error')
    plt.ylabel('Density')

    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(pred_err)

    # Plot the PDF of the fitted normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()

    # Perform normality test (Shapiro-Wilk)
    stat, p_value = stats.shapiro(pred_err)
    print(f'Shapiro-Wilk test statistic: {stat}, p-value: {p_value}')
    if p_value > 0.05:
        print("pred_err appears to be normally distributed (fail to reject H0)")
    else:
        print("pred_err does not appear to be normally distributed (reject H0)")

