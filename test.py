import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.log.lgwr_logger import LgwrLogger


class SpatialDatasetTorch(Dataset):
    def __init__(self, X, y, distance_matrix):
        """
        X: torch.Tensor (n, p) - 自變數
        y: torch.Tensor (n, 1) - 目標變數
        distance_matrix: torch.Tensor (n, n) - 所有點之間的距離矩陣
        """
        self.X = X
        self.y = y
        self.distance_matrix = distance_matrix
        self.n = X.shape[0]

    def __len__(self):
        return self.n  # 總共有 n 個點

    def __getitem__(self, index):
        """
        回傳 (distance_vector, X, y, index) 作為模型的輸入
        """
        distance_vector = self.distance_matrix[index]  # 取出點 i 的距離向量
        xi = self.X[index]  # 取出點 i 的自變數
        yi = self.y[index]  # 取出點 i 的目標變數
        return distance_vector, xi, yi, index  # 必須回傳 index 讓模型知道是哪個點


class LGWR(nn.Module):
    """
    整合 LBNN（學習 bandwidth）、空間加權計算、GWR 回歸的完整 LGWR 模型
    """

    def __init__(self, input_dim, X, y, min_bandwidth=1000, max_bandwidth=10000):
        super(LGWR, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.X = X
        self.y = y
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # LBNN：學習 bandwidth
        self.lbnn = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
            nn.Softplus()
        )

    def forward(self, distance_vector, Xi, yi, index):
        """
        1. 預測 bandwidth
        2. 計算加權矩陣 W
        3. 估計 y_hat
        4. index用於紀錄bandwidth
        """
        # 預測 bandwidth (避免 in-place 操作)
        bandwidth = self.lbnn(distance_vector)
        bandwidth_scoped = bandwidth + 200000

        # bandwidth_scoped = torch.clamp(
        #     bandwidth,
        #     min=self.min_bandwidth, max=self.max_bandwidth
        # )  # 限制範圍

        # if index == 0:
        #     print(
        #         f"{index} | bandwidth: {bandwidth_scoped.item()} | raw: {bandwidth.item()}")

        # 計算空間加權矩陣
        W = self.calculate_weighted_matrix(distance_vector, bandwidth_scoped)

        # 計算 y_hat
        y_hat = self.estimate_y_hat(W, index)
        # if index == 0:
        #     # print(distance_vector)
        #     print(
        #         f"index {index[0]} | y_hat: {y_hat.item()} | bandwidth_scoped: {bandwidth_scoped.item()} | bandwidth: {bandwidth.item()}")

        return y_hat

    def calculate_weighted_matrix(self, distance_vector, bandwidths):
        """
        計算空間加權矩陣 W (exponential)
        """
        weights = torch.exp(- (distance_vector ** 2) / (bandwidths ** 2))
        return weights

    def estimate_y_hat(self, W, index):
        """
        根據加權矩陣 W 估算 y_hat
        """

        XTW = self.X.mT * W
        XTWX = XTW @ self.X
        XTWy = XTW @ self.y
        beta = torch.linalg.solve(XTWX, XTWy)

        # xT = (self.X * W.T).T
        # xtx = torch.matmul(xT, self.X)
        # xtx_inv_xt = torch.linalg.solve(xtx, xT)
        # beta_oldway = torch.matmul(xtx_inv_xt, self.y)

        # 預測 y_hat
        y_hat = torch.matmul(self.X[index], beta)  # (1, p) @ (p, 1) -> (n, 1)

        # if index == 0:
        #     print(y_hat)
        #     print(self.y[index])
        #     print("=========")
        # print(W.shape)
        # print(self.X[index].shape)
        # print(beta.shape)
        # print("=================")

        # if index == 0:
        #     print(f"index {index} | y_hat: {y_hat.item()}")

        return y_hat.reshape(-1, 1)


def train_lgwr(X, y, distance_matrix, epochs=100, lr=0.01, batch_size=1):
    """
    訓練 LGWR，包含 LBNN（學習 bandwidth）與 GWR（計算加權回歸）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpatialDatasetTorch(X, y, distance_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LGWR(
        X.shape[0], X, y
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = 0.0
        y_true_all = torch.empty_like(y)
        y_pred_all = torch.empty_like(y)

        for distance_vector, Xi, yi, index in dataloader:
            optimizer.zero_grad()

            distance_vector = distance_vector.to(device)
            Xi = Xi.to(device)
            yi = yi.to(device)

            yi_hat_batch = model(distance_vector, Xi, yi, index)
            # print(distance_vector.shape)
            # print(Xi.shape)
            # print(yi.shape)
            # print(yi_hat_batch.shape)
            # print("=========")
            # print(distance_vector)
            # print(Xi)
            # print(yi)
            # print(yi_hat_batch)
            # print("=========")

            loss = loss_fn(yi_hat_batch, yi)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()

            # 🚀 儲存所有 batch 的 y_true & y_pred，用於計算 R²
            y_true_all[index] = yi.item()
            y_pred_all[index] = yi_hat_batch.item()

        ss_total = torch.sum((y_true_all - y_true_all.mean()) ** 2)  # 總變異
        ss_residual = torch.sum((y_true_all - y_pred_all) ** 2)  # 殘差變異
        r2_score = 1 - (ss_residual / ss_total)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss} - R2: {r2_score}")

        # print(y_true_all)

        # print(y_pred_all)

    return model


if __name__ == "__main__":

    # 讀取數據
    logger = LgwrLogger()
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],  # X 的欄位
            response_field='PctPov',  # y 的欄位
            coordinate_x_field='X',  # X 座標
            coordinate_y_field='Y'   # Y 座標
        ),
        logger,
        isSpherical=False
    )

    # 轉換為 Torch Tensor
    X = spatialDataset.x_matrix_torch  # (n, p)
    y = spatialDataset.y_torch  # (n, 1)

    # 計算距離矩陣 (n, n)
    distance_matrix = torch.cdist(
        spatialDataset.coordinates_torch, spatialDataset.coordinates_torch, p=2)

    # 訓練模型
    lgwr_model = train_lgwr(
        X, y, distance_matrix,
        epochs=50,  # 訓練 50 個 epochs
        lr=0.001,  # 學習率
        batch_size=32  # 每次更新 1 個點
    )
