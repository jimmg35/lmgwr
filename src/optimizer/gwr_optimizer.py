
# Write a Golden Section Search optimizer for the bandwidth parameter of the GWR model
# The optimizer should be able to find the optimal bandwidth parameter for the GWR model

from typing import Literal, TypeAlias

from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from src.log.gwr_logger import GwrLogger
from src.optimizer.ioptimizer import IOptimizer

GwrBandwidthOptimizeMethod: TypeAlias = Literal['golden_section',
                                                'grid_search', 'random_search']


class GwrOptimizer(IOptimizer):

    method: GwrBandwidthOptimizeMethod
    search_range: tuple

    def __init__(self,
                 model: GWR,
                 kernel: GwrKernel,
                 logger: GwrLogger,
                 method: GwrBandwidthOptimizeMethod = 'golden_section',
                 search_range=(50, 200)) -> None:
        super().__init__(model, kernel, logger)
        self.method = method
        self.search_range = search_range

    def optimize(self):
        if self.method == 'golden_section':
            return self.__golden_section_search(
                self.search_range[0],
                self.search_range[1],
                1.0e-6,
                200
            )
        # elif self.method == 'grid_search':
        #     return self._grid_search()
        # elif self.method == 'deep_learning':
        #     return self._deep_learning_optimizer()
        # else:
        #     raise ValueError(f"未知的方法: {self.method}")

    def __golden_section_search(self, a: float, b: float, tol: float, max_iter: int):
        """
        Perform a golden section search to find the optimal bandwidth parameter for the GWR model.

        Args:
            a (float): The lower bound of the search interval.
            b (float): The upper bound of the search interval.
            tol (float): The tolerance for stopping the search.
            max_iter (int): The maximum number of iterations to perform.

        Returns:
            float: The optimal bandwidth parameter for the GWR model.
        """
        phi = (1 + 5 ** 0.5) / 2

        c = b - (b - a) / phi
        d = a + (b - a) / phi

        fc = self.objective_function(c)
        fd = self.objective_function(d)

        for _ in range(max_iter):
            if fc > fd:
                a = c
                c = d
                d = a + (b - a) / phi
                fc = fd
                fd = self.objective_function(d)
            else:
                b = d
                d = c
                c = b - (b - a) / phi
                fd = fc
                fc = self.objective_function(c)

            if abs(b - a) < tol:
                break

        return (c + d) / 2

    def objective_function(self, bandwidth: float) -> float:
        """
        Calculate the objective function for the GWR model with the given bandwidth.

        Args:
            bandwidth (float): The bandwidth value to optimize.

        Returns:
            float: The AICc value for the GWR model with the given bandwidth.
        """
        self.model.update_bandwidth(bandwidth).fit()
        self.logger.append_bandwidth_optimization(
            f"{self.__class__.__name__} : Bandwidth {bandwidth}, AICc {self.model.aicc}"
        )
        return self.model.aicc
