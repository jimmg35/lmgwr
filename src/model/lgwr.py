import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal

from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.lgwr_logger import LgwrLogger
from src.model.imodel import IModel


class LGWR(IModel):

    optimizeMode: Literal['cuda', 'cpu']

    def __init__(self,
                 dataset: SpatialDataset,
                 kernel: LgwrKernel,
                 logger: LgwrLogger,
                 optimizeMode: Literal['cuda', 'cpu'] = 'cuda') -> None:
        """
        Initializes the GWR model with the specified spatial dataset and kernel.

        Args:
            dataset (SpatialDataset): The spatial dataset containing data points and field information.
            kernel (GwrKernel): A kernel instance used to calculate spatial weights for each data point.
        """

        super().__init__(dataset, kernel, logger)

        self.optimizeMode = optimizeMode

        self.y_hats = torch.zeros(
            self.dataset.x_matrix.shape[0],
            dtype=torch.float32).to(self.optimizeMode)
        self.residuals = torch.zeros(
            self.dataset.x_matrix.shape[0],
            dtype=torch.float32).to(self.optimizeMode)

    def update_local_bandwidth(self, index: int, bandwidth: float | Tensor):
        self.kernel.update_local_bandwidth(index, bandwidth)
        return self

    def _local_fit(self, index: int):
        """
        PyTorch 版本的 `_local_fit`，確保所有矩陣運算都在 PyTorch 上進行
        """
        beta, _, wi = self._estimate_beta_by_index(index)

        # beta = torch.tensor(beta, dtype=torch.float32).to(self.optimizeMode)

        if isinstance(beta, torch.Tensor):
            x_matrix_torch = torch.tensor(
                self.dataset.x_matrix[index, :], dtype=torch.float32).to(self.optimizeMode)
            self.y_hats[index] = torch.matmul(x_matrix_torch, beta)

    def fit(self):
        """
        PyTorch 版本的 `fit()`，確保所有計算都使用 PyTorch Tensor
        """
        for index in range(0, self.dataset.x_matrix.shape[0]):
            self._local_fit(index)

        # ✅ 確保 residuals 也是 PyTorch Tensor
        self.residuals = torch.tensor(
            self.dataset.y,
            dtype=torch.float32
        ).to(self.optimizeMode) - self.y_hats


class LBNN(nn.Module):
    """
    Local Bandwidth Neural Network Model: Input distance vector, output optimal bandwidth
    """

    # input_size: int
    dataset: SpatialDataset
    lgwr: LGWR
    min_bandwidth: int
    max_bandwidth: int

    def __init__(self,
                 dataset: SpatialDataset,
                 lgwr: LGWR,
                 min_bandwidth: int = 10,
                 max_bandwidth: int = 500
                 ):
        super(LBNN, self).__init__()

        self.dataset = dataset
        self.lgwr = lgwr
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        self.fc1 = nn.Linear(self.dataset.x_matrix.shape[0], 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        local_bandwidth = torch.clamp(
            x,
            min=self.min_bandwidth,
            max=self.max_bandwidth
        )

        self.lgwr.update_local_bandwidth(0, local_bandwidth)

        return local_bandwidth
