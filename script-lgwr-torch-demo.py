import pandas as pd

from torch.nn import L1Loss

from src.dataset.spatial_dataset import IFieldInfo
from src.dataset.spatial_dataset_torch import SpatialDataset
from src.model.lgwr_torch import LGWR
from src.log.lgwr_logger import LgwrLogger
from src.utility.optimize_mode import OptimizeMode
from src.optimizer.lgwr_optimizer_torch import LgwrOptimizer


# # 訓練函數
# def train_nn(distance_matrix, X, y, epochs=50, lr=0.01, batch_size=1):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset = DistanceDataset(distance_matrix, y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     model = SimpleNN(distance_matrix.shape[1], X, y).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     for epoch in range(epochs):
#         train_loss = 0.0
#         index = 0
#         y_true_all = torch.empty_like(y)
#         y_pred_all = torch.empty_like(y)

#         for distance_vector, target in dataloader:
#             distance_vector, target = distance_vector.to(
#                 device), target.to(device)

#             optimizer.zero_grad()
#             prediction = model(distance_vector, index)

#             # print(prediction)
#             # print(target)
#             # print(prediction.shape)
#             # print(target.shape)
#             # print("====================")

#             loss = loss_fn(prediction, target)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             y_true_all[index] = target.item()
#             y_pred_all[index] = prediction.item()
#             index += 1

#         ss_total = torch.sum((y_true_all - y_true_all.mean()) ** 2)  # 總變異
#         ss_residual = torch.sum((y_true_all - y_pred_all) ** 2)  # 殘差變異
#         r2_score = 1 - (ss_residual / ss_total)

#         print(
#             f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - R2: {r2_score}")

#     return model

# 測試訓練


# def main():

#     # 讀取數據
#     synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

#     # 轉換為 Tensor
#     X = torch.tensor(synthetic_data[['PctBach', 'PctEld', 'PctBlack']].values,
#                      dtype=torch.float32).to('cuda')
#     y = torch.tensor(synthetic_data['PctPov'].values,
#                      dtype=torch.float32).unsqueeze(1).to('cuda')
#     coordinates = torch.tensor(
#         synthetic_data[['X', 'Y']].values, dtype=torch.float32).to('cuda')

#     # 計算距離矩陣
#     distance_matrix = torch.cdist(coordinates, coordinates, p=2).to('cuda')

#     # print(X.shape)
#     # print(y.shape)
#     # print(coordinates.shape)
#     # print(distance_matrix.shape)

#     # 訓練模型
#     model = train_nn(distance_matrix, X, y)


if __name__ == "__main__":

    # Loading Data
    georgia_dataframe = pd.read_csv(r'./data/GData_utm.csv')
    dataset = SpatialDataset(
        georgia_dataframe,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],
            response_field='PctPov',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        )
    )

    # Initialize components and hyperparameters
    model = LGWR(dataset)
    logger = LgwrLogger()

    loss_function = L1Loss()
    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 1

    # Prepare the training container

    optimizer = LgwrOptimizer(

    )
