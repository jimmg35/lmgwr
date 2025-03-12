import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.log.lgwr_logger import LgwrLogger
# 簡單的 PyTorch Dataset


class DistanceDataset(Dataset):
    def __init__(self, distance_matrix, y):
        self.distance_matrix = distance_matrix  # (n, n)
        self.y = y  # (n, 1)
        self.n = distance_matrix.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.distance_matrix[index], self.y[index]

# 簡單的神經網路回歸模型


class SimpleNN(nn.Module):
    def __init__(self, input_dim, X, y):
        super(SimpleNN, self).__init__()

        self.X = X
        self.y = y
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 預測 y
        )

    def forward(self, distance_vector, index):

        bandwidth = self.model(distance_vector)
        bandwidth_scoped = bandwidth + 200000

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
        XW = self.X * W.view(-1, 1)  # (n, p) * (n, 1) → (n, p)
        XWX = XW.T @ self.X  # (p, n) @ (n, p) → (p, p)
        XWy = XW.T @ self.y  # (p, n) @ (n, 1) → (p, 1)

        # 解線性方程組來得到 beta
        beta = torch.linalg.solve(XWX, XWy)  # (p, p) @ (p, 1) → (p, 1)

        # 使用 index 處的 X 預測 y_hat
        y_hat = torch.matmul(self.X[index], beta)  # (1, p) @ (p, 1) → (1, 1)

        return y_hat.view(1, 1)  # 確保輸出形狀為 (1, 1)


# 訓練函數
def train_nn(distance_matrix, X, y, epochs=50, lr=0.01, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DistanceDataset(distance_matrix, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleNN(distance_matrix.shape[1], X, y).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = 0.0
        index = 0
        for distance_vector, target in dataloader:
            distance_vector, target = distance_vector.to(
                device), target.to(device)

            optimizer.zero_grad()
            prediction = model(distance_vector, index)

            print(prediction)
            print(target)
            print(prediction.shape)
            print(target.shape)
            print("====================")

            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            index += 1

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")

    return model

# 測試訓練


def main():

    # 讀取數據
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

    # 轉換為 Tensor
    X = torch.tensor(synthetic_data[['PctBach', 'PctEld', 'PctBlack']].values,
                     dtype=torch.float32).to('cuda')
    y = torch.tensor(synthetic_data['PctPov'].values,
                     dtype=torch.float32).unsqueeze(1).to('cuda')
    coordinates = torch.tensor(
        synthetic_data[['X', 'Y']].values, dtype=torch.float32).to('cuda')

    # 計算距離矩陣
    distance_matrix = torch.cdist(coordinates, coordinates, p=2).to('cuda')

    # print(X.shape)
    # print(y.shape)
    # print(coordinates.shape)
    # print(distance_matrix.shape)

    # 訓練模型
    model = train_nn(distance_matrix, X, y)


if __name__ == "__main__":
    main()
