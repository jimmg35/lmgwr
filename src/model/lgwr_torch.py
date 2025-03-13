import torch
import torch.nn as nn

from src.dataset.spatial_dataset_torch import SpatialDataset


class LGWR(nn.Module):

    dataset: SpatialDataset

    def __init__(self, dataset: SpatialDataset):
        super(LGWR, self).__init__()
        self.dataset = dataset

        self.model = nn.Sequential(
            nn.Linear(self.dataset.X.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # 預測 y
        )

    def forward(self, distance_vector, index):

        bandwidth = self.model(distance_vector)
        bandwidth_scoped = bandwidth + 200000  # the lower bound of bandwidth is 200000

        # if index == 0:
        #     print(f"{index}: {bandwidth}")

        # adaptive bandwidth
        # 輸出層前加上softmax 並將輸出結果呈上一個常數
        # 讓類神經網路輸出的結果符合adaptive bandwidth的單位
        # 計算空間加權矩陣 W
        W = self.calculate_weighted_matrix(distance_vector, bandwidth_scoped)

        # 計算 y_hat
        y_hat = self.estimate_y_hat(W, index)

        return y_hat

    def calculate_weighted_matrix(self, distance_vector, bandwidths):
        """
        計算空間加權矩陣 W (exponential)
        """
        weights = torch.exp(- (distance_vector ** 2) / (bandwidths ** 2))
        return weights

    def estimate_y_hat(self, W, index):
        """
        計算加權回歸 y_hat (對應單個點)
        """
        # 確保 W 是 (n,) 的 1D 向量
        W = W.squeeze(0)  # 轉換為 (n,)

        # 取出對應的 X, y
        XW = self.dataset.X * W.view(-1, 1)  # (n, p) * (n, 1) → (n, p)
        XWX = XW.T @ self.dataset.X  # (p, n) @ (n, p) → (p, p)
        XWy = XW.T @ self.dataset.y  # (p, n) @ (n, 1) → (p, 1)

        # 解線性方程組來得到 beta
        beta = torch.linalg.solve(XWX, XWy)  # (p, p) @ (p, 1) → (p, 1)

        # 使用 index 處的 X 預測 y_hat
        # (1, p) @ (p, 1) → (1, 1)
        y_hat = torch.matmul(self.dataset.X[index], beta)

        return y_hat.view(1, 1)  # 確保輸出形狀為 (1, 1)
