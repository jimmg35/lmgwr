import numpy as np
import numpy.typing as npt
from scipy import linalg
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel


class GWR:
    """
    A class for performing Geographically Weighted Regression (GWR) using a spatial dataset.

    This class manages the process of fitting a GWR model, which allows for spatially varying
    relationships between predictors and the response variable. It relies on a spatial dataset
    and a kernel for generating location-based weights.

    Attributes:
        dataset (SpatialDataset): The spatial dataset used for the GWR analysis.
        kernel (GwrKernel): The kernel function that defines spatial weights for each location.
    """
    dataset: SpatialDataset
    kernel: GwrKernel

    def __init__(self, dataset: SpatialDataset, kernel: GwrKernel) -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """
        self.dataset = dataset
        self.kernel = kernel

    def fit(self) -> None:
        """
        Fit the GWR model with the provided dataset and spatial weights based on the kernel.

        This method iterates over each data point in the dataset and calculates local regression
        coefficients using spatial weights, implementing the core concept of GWR.

        Raises:
            ValueError: If `dataPoints` are not set up in the dataset.
            NotImplementedError: If the method's fitting logic is not fully implemented.
        """
        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not set up in the dataset")

        # Iterate over each data point to estimate local regression coefficients
        for index in range(len(self.dataset.dataPoints)):
            betas, xtx_inv_xt, wi = self.__estimate_beta_by_index(index)

        raise NotImplementedError("Method not implemented yet")

    def __estimate_beta_by_index(self, index: int):
        """
        Estimates local regression coefficients (betas) for a specified data point.

        This method calculates the coefficients by using weighted least squares regression.
        Spatial weights are obtained from the kernel for the given data point index.

        Args:
            index (int): The index of the data point for which to estimate the coefficients.

        Returns:
            tuple: A tuple containing the local regression coefficients (betas), the matrix (xtx_inv_xt) used
                   to compute the coefficients, and the weight vector (wi) for further analysis.

        Steps:
            1. Obtain the spatial weight vector `wi` for the current data point using the kernel.
            2. Weight the predictor matrix `x_matrix` by `wi` and transpose it to prepare for WLS.
            3. Calculate (X^T * W * X) and its inverse, then multiply by (X^T * W) to solve for betas.
            4. Return the estimated coefficients, the inverse matrix, and the weight vector.

        Raises:
            ValueError: If there is an error in matrix calculations.
        """

        wi = self.kernel.get_weighted_matrix_by_id(index)
        




        wi = self.kernel.get_weighted_matrix_by_id(index)  # Spatial weights for the current data point
        xT = (self.dataset.x_matrix * wi).T  # Weighted predictor matrix transpose (X^T * W)
        xtx = np.dot(xT, self.dataset.x_matrix)  # X^T * W * X
        xtx_inv_xt = linalg.solve(xtx, xT)  # Solve for (X^T * W * X)^-1 * (X^T * W)
        betas = np.dot(xtx_inv_xt, self.dataset.y)  # Compute the local coefficients (betas)
        
        # Return betas, inverse matrix for inspection, and weight vector for trace calculations
        return betas, xtx_inv_xt, wi
