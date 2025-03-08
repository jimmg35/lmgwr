
# Write a Golden Section Search optimizer for the bandwidth parameter of the GWR model
# The optimizer should be able to find the optimal bandwidth parameter for the GWR model

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from typing import Literal, TypeAlias, Dict

GwrBandwidthOptimizeMethod: TypeAlias = Literal['golden_section',
                                                'grid_search', 'random_search']


class GwrBandwidthOptimizer():

    kernel: GwrKernel
    model: GWR
    method: GwrBandwidthOptimizeMethod
    search_range: tuple

    def __init__(self,
                 model: GWR,
                 kernel: GwrKernel,
                 method: GwrBandwidthOptimizeMethod = 'golden_section',
                 search_range=(50, 200)) -> None:
        self.model = model
        self.kernel = kernel
        self.method = method
        self.search_range = search_range

    # def optimize(self):
    #     if self.method == 'golden_section':
    #         return self.__golden_section_search()
    #     # elif self.method == 'grid_search':
    #     #     return self._grid_search()
    #     # elif self.method == 'deep_learning':
    #     #     return self._deep_learning_optimizer()
    #     # else:
    #     #     raise ValueError(f"未知的方法: {self.method}")

    # def __golden_section_search(self, a: float, b: float, tol: float, max_iter: int):
    #     """
    #     Perform a golden section search to find the optimal bandwidth parameter for the GWR model.

    #     Args:
    #         a (float): The lower bound of the search interval.
    #         b (float): The upper bound of the search interval.
    #         tol (float): The tolerance for stopping the search.
    #         max_iter (int): The maximum number of iterations to perform.

    #     Returns:
    #         float: The optimal bandwidth parameter for the GWR model.
    #     """
    #     # Golden ratio
    #     phi = (1 + 5 ** 0.5) / 2

    #     # Initial values
    #     c = b - (b - a) / phi
    #     d = a + (b - a) / phi

    #     # Evaluate the function at the initial points
    #     fc = self.__objective_function(c)
    #     fd = self.__objective_function(d)

    #     # Perform the golden section search
    #     for _ in range(max_iter):
    #         if fc < fd:
    #             b = d
    #             d = c
    #             c = b - (b - a) / phi
    #             fd = fc
    #             fc = self.__objective_function(c)
    #         else:
    #             a = c
    #             c = d
    #             d = a + (b - a) / phi
    #             fc = fd
    #             fd = self.__objective_function(d)

    #         if abs(c - d) < tol:
    #             break

    #     return (c + d) / 2

    # def __objective_function(self, bandwidth: float) -> float:
    #     """
    #     Evaluate the objective function for the GWR model with the given bandwidth parameter.

    #     Args:
    #         bandwidth (float): The bandwidth parameter to evaluate.

    #     Returns:
    #         float: The value of the objective function for the given bandwidth.
    #     """
    #     self.kernel.update_bandwidth(bandwidth)
    #     self.model.fit()
    #     # Calculate the objective function value
    #     # For example, we can use the AICc value as the objective function
    #     aicc = self.model.aicc()
    #     return aicc

    #     # kernel = GwrKernel(self.dataset, bandwidth)
    #     # gwr = GWR(self.dataset, kernel)
    #     # gwr.fit()
    #     # Calculate the objective function value
    #     # For example, we can use the AICc value as the objective function
    #     # aicc = gwr.aicc()
    #     # return aicc
