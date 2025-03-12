import torch
import torch.nn as nn
import torch.optim as optim

# ✅ 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 生成一組簡單的線性數據
torch.manual_seed(42)
X = torch.rand(100, 1, device=device) * 10  # 隨機 100 個 X 值 (0~10)
y = 3 * X + 5 + torch.randn(100, 1, device=device) * 2  # y = 3X + 5 + noise

# ✅ 定義簡單的類神經網路


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 8)  # 1 輸入 -> 8 隱藏層
        self.fc2 = nn.Linear(8, 1)  # 8 隱藏層 -> 1 輸出
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ✅ 初始化模型
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# ✅ 訓練模型
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(X)  # 前向傳播
    loss = loss_fn(y_pred, y)  # 計算損失
    loss.backward()  # 反向傳播
    optimizer.step()  # 更新權重

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ✅ 測試模型
test_X = torch.tensor([[4.0]], device=device)  # 測試 X=4
test_y_pred = model(test_X)
print(f"\n📢 測試 X=4，預測 y={test_y_pred.item():.4f}")
