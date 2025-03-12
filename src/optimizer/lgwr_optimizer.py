import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from typing import Literal, TypeAlias


from src.dataset.spatial_dataset import SpatialDataset
from src.model.imodel import IModel
from src.log.lgwr_logger import LgwrLogger
from src.kernel.lgwr_kernel import LgwrKernel
from src.optimizer.ioptimizer import IOptimizer
from src.model.lgwr import LBNN

# cuda(GPU) or cpu(CPU)
LgwrOptimizeMode: TypeAlias = Literal[
    'cuda', 'cpu'
]


class LgwrOptimizer(IOptimizer):

    lbnn_model: LBNN
    optimizeMode: LgwrOptimizeMode

    def __init__(self,
                 #  model: IModel,
                 lbnn_model: LBNN,
                 kernel: LgwrKernel,
                 logger: LgwrLogger,
                 optimizeMode: LgwrOptimizeMode = 'cuda',
                 lr=0.01,
                 epochs=100
                 ):
        # super().__init__(model, kernel, logger)
        super().__init__(kernel, logger)

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        self.optimizeMode = optimizeMode

        self.logger.append_info("LGWR Optimizer Initialized")
        self.logger.append_info(
            f"{"Using GPU processing :)" if torch.cuda.is_available()
               else "Using CPU processing :("}"
        )

        self.lbnn_model = lbnn_model
        self.lbnn_model.to(self.optimizeMode)
        self.optimizer = optim.Adam(self.lbnn_model.parameters(), lr=lr)
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()

    def optimize(self):
        """
        執行 LGWR 的神經網絡優化
        1. 取得所有點的距離向量，輸入神經網絡
        2. 更新各點的 bandwidth
        3. 訓練模型，並更新神經網絡參數
        """

        if self.lbnn_model.dataset.dataPoints is None:
            raise ValueError("Data points are not initialized in the model.")

        for epoch in range(self.epochs):
            total_loss = 0.0

            for index in range(len(self.lbnn_model.dataset.dataPoints)):
                distance_vector = self.kernel.get_distance_vector_by_id(index)

                self.lbnn_model(distance_vector)

                # 更新 Kernel
                # self.kernel.update_local_bandwidth(
                #     index, predicted_bandwidth
                # )

            # self.model.fit()
            # self.optimizer.zero_grad()

            # print(self.model.dataset.y_torch.shape)
            # print(self.model.y_hats.reshape(-1, 1).shape)
            # print("=================")

            # loss = self.loss_fn(
            #     self.lbnn_model.dataset.y_torch,
            #     self.lbnn_model.lgwr.y_hats.reshape(-1, 1)
            # )

            # before_update = self.lbnn_model.fc1.weight.clone().detach()

            # # 反向傳播
            # loss.backward(retain_graph=True)
            # self.optimizer.step()
            # total_loss += loss.item()

            # after_update = self.lbnn_model.fc1.weight.clone().detach()

            # # 檢查是否有變化
            # print(f"Weight Before: {before_update[:5]}")
            # print(f"Weight After: {after_update[:5]}")
            # print("==================")

            # 記錄 Loss
            # self.logger.append_bandwidth_optimization(
            #     f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}"
            # )

            # , AIC: {self.model.aic}, AICc: {self.model.aicc}, R^2: {self.model.r_squared}

        # return self.model
