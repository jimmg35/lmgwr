import numpy as np
import numpy.typing as npt
from typing import List


class GwrEstimates():

    betas: List[npt.NDArray[np.float64] | None]
    W: List[npt.NDArray[np.float64] | None]
    y_hats
    residules: NDArray[Any]

    # def update_estimates(
    #     self,
    #     beta: npt.NDArray[np.float64],
    #     y_hat: npt.NDArray[np.float64],
    #     residuels: npt.NDArray[np.float64]
    # ):
    #     self.beta = beta
    #     self.y_hat = y_hat
    #     self.residuels = residuels
