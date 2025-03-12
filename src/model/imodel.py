import numpy as np
import numpy.typing as npt
from scipy import linalg
from torch import Tensor
import torch

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.ikernel import IKernel
from src.log.ilogger import ILogger


class IModel:
    dataset: SpatialDataset
    kernel: IKernel
    logger: ILogger

    # estimates for each data point
    betas: npt.NDArray[np.float64] | Tensor
    y_hats: npt.NDArray[np.float64] | Tensor
    S: npt.NDArray[np.float64] | Tensor  # hat matrix
    residuals: npt.NDArray[np.float64] | Tensor

    # matrices for the GWR model
    r_squared: float
    aic: float
    aicc: float

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: IKernel,
                 logger: ILogger) -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """
        self.dataset = dataset
        self.kernel = kernel
        self.logger = logger

        self.logger.append_info(
            f"{self.__class__.__name__} : {self.__class__.__name__} model is initialized.")

    def fit(self) -> None:
        """
        Fit the GWR model with the provided dataset and spatial weights based on the kernel.
        This method iterates over each data point in the dataset and calculates local regression
        estimates for each point.
        """
        raise NotImplementedError("Method not implemented")

    def update_bandwidth(self, bandwidth: float):
        """ GWR"""
        raise NotImplementedError("Method not implemented")

    def update_local_bandwidth(self, index: int, bandwidth: float):
        """ LGWR """
        raise NotImplementedError("Method not implemented")

    def _init_estimates(self) -> None:
        if self.dataset.dataPoints is None:
            raise ValueError(
                "GWR.__init_estimates: DataPoints are not set up in the dataset")

        data_counts = len(self.dataset.dataPoints)

        # allocate memory for the estimates
        self.betas = np.zeros((data_counts, self.dataset.x_matrix.shape[1]))
        self.y_hats = np.zeros(data_counts)
        self.S = np.zeros(data_counts)
        self.residuals = np.zeros(data_counts)

    def _local_fit(self, index: int) -> None:
        """
        Fit a local regression model for a specific data point.

        This method calculates the local regression coefficients (betas) for a given data point
        using weighted least squares regression. The spatial weights are obtained from the kernel
        for the specified data point index.

        Args:
            index (int): The index of the data point for which to estimate the coefficients.

        Raises:
            ValueError: If there is an error in matrix calculations.
        """

        beta, _, wi = self._estimate_beta_by_index(index)

        # calculate elements for estimates and matrices
        XtWX = self.dataset.x_matrix.T @ (wi * self.dataset.x_matrix)
        xi = self.dataset.x_matrix[index, :].reshape(1, -1)
        S_ii = xi @ np.linalg.inv(XtWX) @ xi.T

        # update estimates (in loop to append to the arrays)
        # check type, if numpy array, do the following
        if isinstance(self.betas, np.ndarray):
            self.betas[index, :] = beta.flatten()
        if isinstance(self.y_hats, np.ndarray):
            self.y_hats[index] = self.dataset.x_matrix[index, :] @ beta
        if isinstance(self.S, np.ndarray):
            self.S[index] = S_ii.flatten()[0]
        # self.betas[index, :] = beta.flatten()
        # self.y_hats[index] = self.dataset.x_matrix[index, :] @ beta
        # self.S[index] = S_ii.flatten()[0]

    def _estimate_beta_by_index(self, index: int):
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

        wi: npt.NDArray[np.float64] | Tensor = self.kernel.get_weighted_matrix_by_id(
            index)

        if isinstance(wi, Tensor):
            wi = wi.to('cuda')

            X = self.dataset.x_matrix_torch
            y = self.dataset.y_torch

            # ✅ PyTorch 版本的加權矩陣計算
            xT = (X * wi).T  # `wi` 是 (n,1)，這樣做會廣播
            xtx = torch.matmul(xT, X)  # (p, n) @ (n, p) -> (p, p)

            # ✅ 解線性方程組以獲得 `xtx_inv_xt`
            # PyTorch 版本的 `linalg.solve()`
            xtx_inv_xt = torch.linalg.solve(xtx, xT)

            print(type(xtx_inv_xt))

            # ✅ 計算 `beta`
            beta = torch.matmul(xtx_inv_xt, y)  # (p, n) @ (n, 1) -> (p, 1)

            return beta, xtx_inv_xt, wi
        else:
            xT = (self.dataset.x_matrix * wi).T
            xtx = np.dot(xT, self.dataset.x_matrix)
            xtx_inv_xt: npt.NDArray[np.float64] = linalg.solve(xtx, xT)
            beta: npt.NDArray[np.float64] = np.dot(xtx_inv_xt, self.dataset.y)

            # Return betas, inverse matrix for inspection, and weight vector for trace calculations
            # betas:      (number of independent vars, 1)
            # xtx_inv_xt: (number of independent vars, number of datapoints)
            # wi:         (number of datapoints, 1)
            return beta, xtx_inv_xt, wi

    def _calculate_r_squared(self) -> None:
        """
        Calculate the R-squared value for the GWR model.

        The R-squared value is a measure of the model's goodness of fit, indicating the proportion
        of variance in the response variable that is explained by the predictors.

        Raises:
            NotImplementedError: If the method is not fully implemented.
        """
        y_bar = np.mean(self.dataset.y)
        ss_total = np.sum((self.dataset.y - y_bar) ** 2)
        ss_res = np.sum(self.residuals ** 2)
        self.r_squared = float(1 - ss_res / ss_total)
        self.logger.update_matrics('R-squared', self.r_squared)

    def _calculate_aic_aicc(self) -> None:
        """
        Calculate the Akaike Information Criterion (AIC) for the GWR model.

        The AIC is a measure of the model's goodness of fit, balancing the likelihood of the model
        with the number of parameters used. Lower AIC values indicate better models.

        Returns:
            float: The AIC value for the GWR model.

        Raises:
            NotImplementedError: If the method is not fully implemented.
        """

        if self.dataset.dataPoints is None:
            raise ValueError("DataPoints are not set up in the dataset")

        n = len(self.dataset.dataPoints)
        RSS = np.sum(self.residuals ** 2)
        sigma2 = RSS / n
        trS = np.sum(self.S)
        llf = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1)
        AIC = -2.0 * llf + 2.0 * (trS + 1)
        AICc = AIC + (2.0 * trS * (trS + 1.0)) / (n - trS - 2.0)
        self.aic = AIC
        self.aicc = AICc

        self.logger.update_matrics('AIC', self.aic)
        self.logger.update_matrics('AICc', self.aicc)
