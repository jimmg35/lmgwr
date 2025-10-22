import numpy as np
import numpy.typing as npt

from src.model.base import Base
from src.model.lgwr import LGWR
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.lgwr_kernel import LgwrKernel


class LMGWR(Base):
    """
    Local Multiscale Geographically Weighted Regression (LMGWR).

    Each observation `i` is associated with its own bandwidth set over the `k` covariates,
    enabling location-specific multi-scale smoothing when estimating local coefficients.
    """

    bandwidth_matrix: npt.NDArray[np.float64] | None

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: LgwrKernel) -> None:
        super().__init__(dataset, kernel)
        self.model_type = "LMGWR"
        self.bandwidth_matrix = None

    def exact_fit(self) -> "LMGWR":
        """
        Fit the LMGWR model using the supplied per-observation bandwidth matrix.

        Raises:
            TypeError: If the kernel is not an instance of `LgwrKernel`.
            ValueError: If `bandwidth_matrix` is not provided or has incompatible shape.
        """
        if not isinstance(self.kernel, LgwrKernel):
            raise TypeError("Kernel must be of type LgwrKernel for LMGWR.")

        if self.bandwidth_matrix is None:
            raise ValueError("bandwidth_matrix must be provided before fitting the model.")

        if self.bandwidth_matrix.shape != (self.dataset.n, self.dataset.k):
            raise ValueError(
                "bandwidth_matrix must have shape (n, k) matching the dataset dimensions."
            )

        P_blocks: list[list[np.ndarray]] = []
        Q_blocks: list[list[np.ndarray]] = []
        identity = np.eye(self.dataset.n)

        # Iterate over each covariate to assemble the block matrices.
        for column_index in range(self.dataset.k):
            self.dataset.use_column(column_index)

            local_kernel = LgwrKernel(
                self.dataset,
                kernel_type=self.kernel.kernel_type,
                kernel_bandwidth_type=self.kernel.kernel_bandwidth_type
            )

            local_model = LGWR(self.dataset, local_kernel)
            local_bandwidths = self.bandwidth_matrix[:, column_index]

            local_model.update_local_bandwidth_vector(local_bandwidths).fit()
            Aj = local_model.S

            row_blocks = []
            for inner_index in range(self.dataset.k):
                row_blocks.append(identity if column_index == inner_index else Aj)

            P_blocks.append(row_blocks)
            Q_blocks.append([Aj])

        P = np.block(P_blocks)
        Q = np.block(Q_blocks)
        R = np.linalg.solve(P, Q)
        fitted = R.dot(self.dataset.y)

        self.dataset.reset_columns()
        params = fitted / self.dataset.X.T.reshape(-1, 1)
        params = params.reshape(-1, self.dataset.n).T

        R_tensor = np.stack(np.split(R, self.dataset.k), axis=2)
        ENP_j = np.trace(R_tensor, axis1=0, axis2=1)
        predy = np.sum(self.dataset.X * params, axis=1).reshape(-1, 1)

        self.S = np.sum(R_tensor, axis=2)
        self.y_hats = predy
        self.betas = params
        self.ENP_j = ENP_j

        super()._calculate_residuals()
        super()._calculate_mu()
        super()._calculate_llf()
        super()._calculate_tr_S()
        super()._calculate_r_squared()
        super()._calculate_aic_aicc()

        return self

    def update_bandwidth_matrix(self,
                                bandwidth_matrix: npt.ArrayLike) -> "LMGWR":
        matrix = np.asarray(bandwidth_matrix, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError("bandwidth_matrix must be a 2-D array.")

        if matrix.shape != (self.dataset.n, self.dataset.k):
            raise ValueError(
                "bandwidth_matrix must have shape (n, k) matching the dataset dimensions."
            )

        if np.any(matrix <= 0):
            raise ValueError("bandwidth_matrix values must be positive.")

        self.bandwidth_matrix = matrix
        return self

