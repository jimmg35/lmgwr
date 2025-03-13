
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

from src.dataset.spatial_dataset_torch import SpatialDataset


class LGWR(nn.Module):

    dataset: SpatialDataset
    local_bandwidths: Dict[int, Tensor] = {}
    local_coefficients: Dict[int, Tensor] = {}

    def __init__(self, dataset: SpatialDataset):
        super(LGWR, self).__init__()
        self.dataset = dataset

        self.lbnn = nn.Sequential(  # self.dataset.distance_matrix.shape[0]
            nn.Linear(self.dataset.X.shape[1], 512),
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
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        ).to(self.dataset.optimizeMode)

    def forward(self, distance_vector: Tensor, index: int):
        bandwidth = self.lbnn(distance_vector)
        return bandwidth

        # W = self.calculate_weighted_matrix(distance_vector, bandwidth)
        # y_hat = self.estimate_y_hat(W, index)
        # return y_hat

        # bandwidth_scoped = bandwidth * distance_vector.max().detach()
        # W = self.calculate_weighted_matrix(distance_vector, bandwidth_scoped)
        # return W.mean()
        # y_hat = self.estimate_y_hat(W, index)
        # self.local_bandwidths[index] = bandwidth_scoped
        # return y_hat

    def calculate_weighted_matrix(self, distance_vector, bandwidth):
        """
        calculate the weighted matrix
        """
        zs = distance_vector / bandwidth
        weights = torch.exp(-zs)
        return weights

    def estimate_y_hat(self, W, index):
        """
        estimate the y_hat
        """
        W = W.squeeze(0)

        XW = self.dataset.X * W.view(-1, 1)  # (n, p) * (n, 1) → (n, p)
        XWX = XW.T @ self.dataset.X  # (p, n) @ (n, p) → (p, p)
        XWy = XW.T @ self.dataset.y  # (p, n) @ (n, 1) → (p, 1)

        beta: Tensor = torch.linalg.solve(XWX, XWy)  # (p, p) @ (p, 1) → (p, 1)
        self.local_coefficients[index] = beta

        # (1, k) @ (k, 1) → (1, 1)
        y_hat = torch.matmul(self.dataset.X[index], beta)

        return y_hat.view(1, 1)

    def monitor_layer_weights(self):
        fc1_weights = self.lbnn[4].weight
        print(fc1_weights)
