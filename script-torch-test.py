import torch
import torch.nn as nn
import torch.optim as optim


class LGWR(nn.Module):
    """
    整合 LBNN、空間加權計算、回歸運算的完整 LGWR 模型
    """

    def __init__(self, X, y, coordinates, min_bandwidth=10, max_bandwidth=500):
        super(LGWR, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 訓練數據
        self.X = X.to(self.device)  # (n, p)
        self.y = y.to(self.device)  # (n, 1)
        self.coordinates = coordinates.to(self.device)  # (n, 2)

        self.n = self.X.shape[0]  # 樣本數
        self.p = self.X.shape[1]
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # LBNN：用於預測每個點的 bandwidth
        self.lbnn = nn.Sequential(
            nn.Linear(self.p, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # 確保 bandwidth > 0
        ).to(self.device)

    def forward(self):
        """
        1. 計算每個點的 bandwidth
        2. 使用 bandwidth 更新空間加權矩陣
        3. 計算每個點的局部回歸係數，得到 y_hat
        """
        # # 計算所有點的距離矩陣 (n, n)
        # distance_matrix = torch.cdist(self.coordinates, self.coordinates, p=2)

        # # 預測 bandwidth (n, 1)
        # bandwidths = self.lbnn(distance_matrix)
        # bandwidths = torch.clamp(
        #     bandwidths, min=self.min_bandwidth, max=self.max_bandwidth)  # 限制範圍

        # # 初始化空間加權矩陣 (n, n)
        # weighted_matrix = self.calculate_weighted_matrix(
        #     distance_matrix, bandwidths)

        # # 計算 y_hat
        # y_hat = self.estimate_y_hat(weighted_matrix)

        # return y_hat, bandwidths

        z = self.lbnn(self.X)
        z += 8
        return z

    def calculate_weighted_matrix(self, distance_matrix, bandwidths):
        """
        計算空間加權矩陣 W
        """
        n = self.n
        W = torch.zeros((n, n), device=self.device)

        for i in range(n):
            # 計算該點的加權權重 (Gaussian Kernel)
            W[i, :] = torch.exp(-0.5 *
                                (distance_matrix[i, :] / bandwidths[i]) ** 2)

        return W

    def estimate_y_hat(self, W):
        """
        根據加權矩陣 W 估算 y_hat
        """
        XTW = torch.matmul(self.X.T, W)  # (p, n) @ (n, n) -> (p, n)
        XTWX = torch.matmul(XTW, self.X)  # (p, n) @ (n, p) -> (p, p)
        XTWy = torch.matmul(XTW, self.y)  # (p, n) @ (n, 1) -> (p, 1)

        # 求解 β = (XTWX)^-1 * XTWy
        beta = torch.linalg.solve(XTWX, XTWy)  # (p, p) * (p, 1) -> (p, 1)

        # 預測 y_hat
        y_hat = torch.matmul(self.X, beta)  # (n, p) @ (p, 1) -> (n, 1)

        return y_hat


# 訓練 LBNN + LGWR
def train_lgwr(X, y, coordinates, epochs=100, lr=0.01):
    """
    訓練 LGWR，包含 LBNN（學習 bandwidth）與 GWR（計算加權回歸）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LGWR(X, y, coordinates).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_hat = model()
        loss = loss_fn(y_hat, model.y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

    return model


if __name__ == '__main__':
    # 生成假數據
    n = 100  # 樣本數
    p = 3  # 特徵數

    X = torch.randn(n, p, dtype=torch.float32)  # 隨機 X 矩陣
    y = torch.randn(n, 1, dtype=torch.float32)  # 隨機 y
    coordinates = torch.rand(n, 2, dtype=torch.float32) * 100  # 生成隨機座標

    # 訓練 LGWR
    lgwr_model = train_lgwr(X, y, coordinates, epochs=50, lr=0.01)
