import numpy as np
import numpy.typing as npt
from gstools import SRF, CovModel
import matplotlib.pyplot as plt
from matplotlib import colormaps

# from src.dataset.interfaces.idataset import IDataset
from src.dataset.spatial_dataset import SpatialDataset


class SimulatedSpatialDataset(SpatialDataset):

    field_size: int
    data_seed: int
    process_seed: list[int]
    k: int
    coordinates: npt.NDArray[np.float64]

    def __init__(self,
                 field_size=40,
                 data_seed=222,
                 process_seed=[555, 888],
                 k=1
                 ) -> None:
        self.field_size = field_size
        self.data_seed = data_seed
        self.process_seed = process_seed
        self.k = k

        self.__generate_data()

    def __generate_data(self):
        np.random.seed(self.data_seed)
        X_list = []
        for _ in range(self.k + 1):
            X_list.append(np.random.randn(self.field_size * self.field_size))
        self.X = np.vstack(X_list).T
        # Add a column of ones as the first column for the intercept
        if self.useIntercept:
            self.X = np.hstack(
                (np.ones((self.X.shape[0], 1)), self.X)
            )

        u = np.array([np.linspace(0, self.field_size-1,
                     num=self.field_size)]*self.field_size).reshape(-1)
        v = np.array([np.linspace(0, self.field_size-1,
                     num=self.field_size)]*self.field_size).T.reshape(-1)
        self.coordinates = np.array(list(zip(u, v)))

    def generate_processes(self):
        np.random.seed(self.data_seed)

        class GWR_gau(CovModel):
            def correlation(self, r):
                return np.exp(-0.5 * (r / self.len_scale) ** 2)

        coords = [
            range(self.field_size),
            range(self.field_size)
        ]

        processes = []
        for i, seed in enumerate(self.process_seed):
            model = GWR_gau(dim=2, var=1, len_scale=6 * (i + 1))
            srf = SRF(model, mean=0, seed=seed)
            process = srf.structured(coords).reshape(-1)
            process = (process - process.mean()) / process.std() + 2
            processes.append(process)

        b2 = np.ones(self.field_size * self.field_size).reshape(-1) * 2
        processes.append(b2)

        return processes

    def fit_y(self, b0, b1, b2):
        err = np.random.randn(self.field_size * self.field_size)
        self.y = (b0 * self.X[:, 0] + b1 * self.X[:, 1] + b2 *
                  self.X[:, 2] + err).reshape(-1, 1)
        return [self.X, self.y]

    def plot(self, b, sub_title=['', '', '', ''], size=40, vmin=None, vmax=None):
        k = len(b)
        fig, axs = plt.subplots(1, k, figsize=(6*k, 4))
        for i in range(k):
            if i == 0:
                ax = axs[i].imshow(b[i].reshape(size, size),
                                   cmap=colormaps['viridis'], vmin=vmin, vmax=vmax)
            else:  # plt.cm.get_cmap('viridis', 21)
                ax = axs[i].imshow(b[i].reshape(size, size),
                                   cmap=colormaps['viridis'], vmin=vmin, vmax=vmax)
            axs[i].set_title(sub_title[i], fontsize=16)
            fig.colorbar(ax, ax=axs[i])

            axs[i].set_xticks(np.arange(-0.5, 40, 5))
            axs[i].set_yticks(np.arange(-0.5, 40, 5))
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])

            axs[i].tick_params(axis='x', colors=(0, 0, 0, 0))
            axs[i].tick_params(axis='y', colors=(0, 0, 0, 0))
        plt.show()
