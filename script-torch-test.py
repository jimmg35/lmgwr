import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.log.lgwr_logger import LgwrLogger


class LGWR(nn.Module):
    """
    整合 LBNN、空間加權計算、回歸運算的完整 LGWR 模型
    """

    def __init__(self, X, y, min_bandwidth=10, max_bandwidth=500):
        super(LGWR, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 訓練數據
        self.X = X  # (n, p)
        self.y = y  # (n, 1)
        # self.coordinates = coordinates  # (n, 2)

        self.n = self.X.shape[0]  # 樣本數
        self.p = self.X.shape[1]

        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # LBNN：用於預測每個點的 bandwidth
        self.lbnn = nn.Sequential(
            nn.Linear(self.n, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # 確保 bandwidth >= 0
        ).to(self.device)

    def forward(self, distance_vector, index):
        """
        1. 計算每個點的 bandwidth
        2. 使用 bandwidth 更新空間加權矩陣
        3. 計算每個點的局部回歸係數，得到 y_hat
        """

        # 預測 bandwidth (n, 1)
        bandwidth = self.lbnn(distance_vector)
        bandwidth = bandwidth + 100
        # bandwidth_scoped = torch.clamp(
        #     bandwidth,
        #     min=self.min_bandwidth, max=self.max_bandwidth
        # )  # 限制範圍

        # # 初始化空間加權矩陣 (n, n)
        weighted_matrix = self.calculate_weighted_matrix(
            distance_vector, bandwidth)

        # 計算 y_i_hat
        y_i_hat = self.estimate_y_i_hat(weighted_matrix)

        return y_i_hat[index]

    def calculate_weighted_matrix(self, distance_vector, bandwidths):
        """
        計算空間加權矩陣 W
        """

        # 確保 bandwidths 形狀匹配 distance_vector
        if bandwidths.shape != distance_vector.shape:
            bandwidths = bandwidths.view(1).expand_as(distance_vector)

        weights = torch.exp(- (distance_vector ** 2) / (bandwidths ** 2))
        return weights

    def estimate_y_i_hat(self, W):
        """
        根據加權矩陣 W 估算 y_i_hat
        """
        # 直接對 X.T 進行逐元素相乘，避免創建對角矩陣
        XTW = self.X.T * W  # (p, n) * (n,) -> (p, n)
        XTWX = torch.matmul(XTW, self.X)  # (p, n) @ (n, p) -> (p, p)
        XTWy = torch.matmul(XTW, self.y)  # (p, n) @ (n, 1) -> (p, 1)

        # 求解 β = (XTWX)^-1 * XTWy
        beta = torch.linalg.solve(XTWX, XTWy)  # (p, p) * (p, 1) -> (p, 1)

        # 預測 y_hat
        y_i_hat = torch.matmul(self.X, beta)  # (n, p) @ (p, 1) -> (n, 1)

        return y_i_hat


# 訓練 LBNN + LGWR
def train_lgwr(X, y, distance_matrix, epochs=100, lr=0.01):
    """
    訓練 LGWR，包含 LBNN（學習 bandwidth）與 GWR（計算加權回歸）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LGWR(X, y).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = 0.0

        # i need the distance vector and its index
        for index, distance_vector in enumerate(distance_matrix):
            if index == 0:
                optimizer.zero_grad()

                y_i_hat = model(distance_vector, index)

                print("========================")
                print(y_i_hat)
                print(model.y[index])

                loss = loss_fn(y_i_hat, model.y[index])

                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss}")

    return model


if __name__ == '__main__':

    logger = LgwrLogger()
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')
    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],
            response_field='PctPov',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        logger,
        isSpherical=False
    )

    # 計算所有點的距離矩陣 (n, n)
    distance_matrix = torch.cdist(
        spatialDataset.coordinates_torch, spatialDataset.coordinates_torch, p=2)

    # 訓練 LGWR
    lgwr_model = train_lgwr(spatialDataset.x_matrix_torch, spatialDataset.y_torch,
                            distance_matrix, epochs=50, lr=0.01)
