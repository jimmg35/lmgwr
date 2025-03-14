
import torch
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader

from src.model.lgwr_torch import LGWR
from src.log.lgwr_logger import LgwrLogger
from src.utility.optimize_mode import OptimizeMode
from src.dataset.spatial_dataset_torch import SpatialDataset


class LgwrOptimizer():

    model: LGWR
    logger: LgwrLogger
    optimizeMode: torch.device

    # hyperparameters
    learning_rate: float
    epochs: int
    batch_size: int

    # Torch components
    dataset: SpatialDataset
    dataLoader: DataLoader
    loss_function: MSELoss | L1Loss

    def __init__(self,
                 model: LGWR,
                 logger: LgwrLogger,
                 dataset: SpatialDataset,
                 loss_function: MSELoss | L1Loss,
                 optimizeMode: torch.device,
                 learning_rate=0.01,
                 epochs=100,
                 batch_size=1
                 ):

        self.model = model
        self.logger = logger
        self.dataset = dataset
        self.loss_function = loss_function
        self.optimizeMode = optimizeMode

        # hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # torch training components
        self.dataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.logger.append_info("LGWR Optimizer Initialized")
        self.logger.append_info(
            f"{"Using GPU processing :)" if self.optimizeMode == 'cuda'
               else "Using CPU processing :("}"
        )

    def train(self):
        # for name, param in self.model.lbnn.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")

        # print("+=========================")
        # for name, param in self.model.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")
        # print("+=========================")
        # self.model.monitor_layer_weights()

        for epoch in range(self.epochs):
            train_loss = 0.0
            index = 0
            y_true_all = torch.empty_like(self.dataset.y)
            y_pred_all = torch.empty_like(self.dataset.y)

            for distance_vector_batch, xi_batch, yi_batch in self.dataLoader:
                # get a batch of data (usually minibatch)
                distance_vector_batch: torch.Tensor = distance_vector_batch.to(
                    self.optimizeMode)
                xi_batch: torch.Tensor = xi_batch.to(self.optimizeMode)
                yi_batch: torch.Tensor = yi_batch.to(self.optimizeMode)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                yi_hat_batch = self.model(xi_batch, index)

                # print(f"yi_hat_batch: {yi_hat_batch.item()}")
                # print(f"yi_batch: {yi_batch.item()}")

                loss = self.loss_function(yi_hat_batch, yi_batch)

                # print(f"Loss before backward: {loss.item()}")
                # print("=====================")

                loss.backward()
                # print("After backward:")
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         # 梯度應該 > 0
                #         print(
                #             f"{name} grad norm: {torch.norm(param.grad).item():.8f}")

                self.optimizer.step()
                train_loss += loss.item()

                y_true_all[index] = yi_batch.item()
                y_pred_all[index] = yi_hat_batch.item()

                index += 1

            ss_total = torch.sum((y_true_all - y_true_all.mean()) ** 2)
            ss_residual = torch.sum((y_true_all - y_pred_all) ** 2)
            r2_score = 1 - (ss_residual / ss_total)

            print(
                f"Epoch {epoch+1}/{self.epochs} | Loss: {train_loss:.4f} - R2: {r2_score}"
            )

            # self.model.monitor_layer_weights()
            # print(self.model.local_bandwidths)

    # def predict(self, x):
    #     x = torch.tensor(x, dtype=torch.float32).to(self.optimizeMode)
    #     bandwidth = self.lbnn_model(x)
    #     return bandwidth.detach().cpu().numpy()

    # def save(self, path):
    #     torch.save(self.lbnn_model.state_dict(), path)

    # def load(self, path):
    #     self.lbnn_model.load_state_dict(torch.load(path))
    #     self.lbnn_model.eval()
