
# Write a Golden Section Search optimizer for the bandwidth parameter of the GWR model
# The optimizer should be able to find the optimal bandwidth parameter for the GWR model

import logging
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from typing import Literal, TypeAlias, Dict
from src.log.logger import GwrLogger

GwrBandwidthOptimizeMethod: TypeAlias = Literal['golden_section',
                                                'grid_search', 'random_search']


class GwrBandwidthOptimizer():

    kernel: GwrKernel
    model: GWR
    logger: GwrLogger

    method: GwrBandwidthOptimizeMethod
    search_range: tuple

    def __init__(self,
                 model: GWR,
                 kernel: GwrKernel,
                 logger: GwrLogger,
                 method: GwrBandwidthOptimizeMethod = 'golden_section',
                 search_range=(50, 200)) -> None:
        self.model = model
        self.kernel = kernel
        self.logger = logger
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
        # 黃金比例
        phi = (1 + 5 ** 0.5) / 2

        # 初始化區間
        c = b - (b - a) / phi
        d = a + (b - a) / phi

        # 計算初始 AICc
        fc = self.__objective_function(c)
        fd = self.__objective_function(d)

        for _ in range(max_iter):
            # 目標是最小化 AICc
            if fc > fd:  # ✅ 這裡要改成 ">"，因為我們希望 AICc 越小越好
                a = c
                c = d
                d = a + (b - a) / phi
                fc = fd
                fd = self.__objective_function(d)
            else:
                b = d
                d = c
                c = b - (b - a) / phi
                fd = fc
                fc = self.__objective_function(c)

            # 終止條件：當區間長度小於 tol 時
            if abs(b - a) < tol:
                break

        # 返回最佳帶寬
        return (c + d) / 2

    def __objective_function(self, bandwidth: float) -> float:
        """
        Evaluate the objective function for the GWR model with the given bandwidth parameter.

        Args:
            bandwidth (float): The bandwidth parameter to evaluate.

        Returns:
            float: The value of the objective function for the given bandwidth.
        """
        self.kernel.update_bandwidth(bandwidth)
        self.model.fit()

        self.logger.append_bandwidth_optimization(
            f"GwrBandwidthOptimizer : Bandwidth {bandwidth}, AICc {self.model.aicc}"
        )
        return self.model.aicc
