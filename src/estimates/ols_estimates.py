import numpy as np
import numpy.typing as npt


class OlsEstimates():

    beta: npt.NDArray[np.float64] | None
    y_hat: npt.NDArray[np.float64] | None
    residuels: npt.NDArray[np.float64] | None

    def update_estimates(
        self,
        beta: npt.NDArray[np.float64],
        y_hat: npt.NDArray[np.float64],
        residuels: npt.NDArray[np.float64]
    ):
        self.beta = beta
        self.y_hat = y_hat
        self.residuels = residuels
