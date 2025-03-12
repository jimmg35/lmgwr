import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ✅ 1️⃣ 定義 `SpatialDataset`
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
        return distance_vector, self.X, self.y, index  # 回傳 index 讓模型知道是哪個點


# ✅ 2️⃣ 定義 `LGWR` 模型
class LGWR(nn.Module):
    """
    將 GWR 運算整合進 LBNN，讓 PyTorch 自動管理計算圖
    """

    def __init__(self, input_dim, feature_dim):
        super(LGWR, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # **LBNN 預測 bandwidth**
        self.lbnn = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # 確保 bandwidth >= 0
        )

        # **GWR 參數回歸層**（LBNN 直接輸出 `beta_i` 而不是 bandwidth）
        self.gwr_regressor = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, distance_vector, X, y, index):
        """
        1. 由 LBNN 預測 bandwidth
        2. 直接在 forward() 計算 GWR 權重 & beta_i
        3. 返回 `y_hat`
        """

        # **1️⃣ LBNN 預測 bandwidth**
        bandwidth = self.lbnn(distance_vector)
        bandwidth = bandwidth + 100  # 避免 0 值

        # **2️⃣ 計算加權矩陣 W**
        W = torch.exp(- (distance_vector ** 2) / (bandwidth ** 2))

        # **3️⃣ 計算 GWR 的 β_i**
        XTW = X.mT * W.unsqueeze(0)  # (p, n) * (1, n) -> (p, n)
        XTWX = torch.matmul(XTW, X)  # (p, n) @ (n, p) -> (p, p)
        XTWy = torch.matmul(XTW, y)  # (p, n) @ (n, 1) -> (p, 1)

        # **求解 β_i**
        beta_i = torch.linalg.solve(XTWX, XTWy)  # (p, p) * (p, 1) -> (p, 1)

        # **4️⃣ 使用 `gwr_regressor` 訓練 β_i**
        beta_i = self.gwr_regressor(beta_i.T).T  # (p, 1)

        # **5️⃣ 預測 `y_hat`**
        y_hat = torch.matmul(X, beta_i)  # (n, p) @ (p, 1) -> (n, 1)

        return y_hat[index]  # 只取出該點的 y_hat


# ✅ 3️⃣ 訓練 `LGWR`
def train_lgwr(X, y, distance_matrix, epochs=100, lr=0.01, batch_size=1):
    """
    訓練 LGWR，包含 LBNN（學習 bandwidth）與 GWR（計算加權回歸）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpatialDatasetTorch(X, y, distance_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LGWR(input_dim=X.shape[0], feature_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = 0.0

        for distance_vector, X_batch, y_batch, index in dataloader:
            optimizer.zero_grad()

            distance_vector = distance_vector.to(device)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_i_hat = model(distance_vector, X_batch, y_batch, index)

            loss = loss_fn(y_i_hat, y_batch.squeeze())  # 確保 batch loss 正確

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss}")

    return model


# ✅ 4️⃣ 讀取數據 & 計算距離矩陣
if __name__ == '__main__':
    # 讀取數據
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

    # 創建 SpatialDataset
    from src.dataset.spatial_dataset import SpatialDataset
    from src.dataset.interfaces.spatial_dataset import IFieldInfo
    from src.log.lgwr_logger import LgwrLogger

    logger = LgwrLogger()
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

    # 訓練 LGWR
    lgwr_model = train_lgwr(
        X, y, distance_matrix,
        epochs=50,  # 訓練 50 個 epochs
        lr=0.01,  # 學習率
        batch_size=1  # 每次更新 1 個點
    )

    # 測試模型（預測某個點的 `y_hat`）
    i = 10
    distance_vector = distance_matrix[i].unsqueeze(0).to(lgwr_model.device)
    y_hat_i = lgwr_model(distance_vector, X.to(
        lgwr_model.device), y.to(lgwr_model.device), i)

    print(f"第 {i} 個點的預測值: {y_hat_i.item()}")
