import torch
import torch.nn as nn
import torch.optim as optim
from typing import Literal, TypeAlias

from src.dataset.spatial_dataset import SpatialDataset
from src.model.imodel import IModel
from src.log.lgwr_logger import LgwrLogger
from src.kernel.lgwr_kernel import LgwrKernel
from src.optimizer.ioptimizer import IOptimizer

# cuda(GPU) or cpu(CPU)
LgwrOptimizeMode: TypeAlias = Literal[
    'cuda', 'cpu'
]


class LBNN(nn.Module):
    """
    Local Bandwidth Neural Network Model: Input distance vector, output optimal bandwidth
    """

    # input_size: int
    dataset: SpatialDataset
    min_bandwidth: int
    max_bandwidth: int

    def __init__(self,
                 dataset: SpatialDataset,
                 #  input_size: int,
                 min_bandwidth: int = 10,
                 max_bandwidth: int = 500
                 ):
        super(LBNN, self).__init__()

        self.dataset = dataset
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        self.fc1 = nn.Linear(self.dataset.x_matrix.shape[0], 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # 輸出單一帶寬
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        bandwidth = torch.nn.functional.softplus(x)  # 保證 > 0
        bandwidth = torch.clamp(
            bandwidth,
            min=self.min_bandwidth,
            max=self.max_bandwidth
        )  # 限制範圍
        return bandwidth


class LgwrOptimizer(IOptimizer):

    lbnn_model: LBNN
    optimizeMode: LgwrOptimizeMode

    def __init__(self,
                 model: IModel,
                 lbnn_model: LBNN,
                 kernel: LgwrKernel,
                 logger: LgwrLogger,
                 optimizeMode: LgwrOptimizeMode = 'cuda',
                 lr=0.01,
                 epochs=100
                 ):
        super().__init__(model, kernel, logger)

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

        if self.model.dataset.dataPoints is None:
            raise ValueError("Data points are not initialized in the model.")

        for epoch in range(self.epochs):
            total_loss = 0.0

            for index in range(len(self.model.dataset.dataPoints)):
                distance_vector = self.kernel.get_distance_vector_by_id(index)
                distance_vector = torch.tensor(
                    distance_vector, dtype=torch.float32
                ).to(self.optimizeMode)

                # 預測 bandwidth
                predicted_bandwidth = self.lbnn_model(distance_vector)

                # if index == 0 or index == 1 or index == 2:
                #     print(f"{index}, {predicted_bandwidth.item()}")

                # 更新 Kernel
                self.kernel.update_local_bandwidth(
                    index, predicted_bandwidth.item()
                )

            self.model.fit()

            y_true = torch.tensor(
                self.model.dataset.y,
                dtype=torch.float32,
                requires_grad=True
            ).to(self.optimizeMode)

            y_hat = torch.tensor(
                self.model.y_hats.reshape(-1, 1),
                dtype=torch.float32,
                requires_grad=True
            ).to(self.optimizeMode)

            loss = self.loss_fn(y_true, y_hat)

            # before_update = self.lbnn_model.fc1.weight.clone().detach()

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # after_update = self.lbnn_model.fc1.weight.clone().detach()

            # # 檢查是否有變化
            # print(f"Weight Before: {before_update[:5]}")
            # print(f"Weight After: {after_update[:5]}")
            # print("==================")

            # 記錄 Loss
            self.logger.append_bandwidth_optimization(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}, AIC: {self.model.aic}, AICc: {self.model.aicc}, R^2: {self.model.r_squared}"
            )

        return self.model
