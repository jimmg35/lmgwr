import numpy as np
import numpy.typing as npt
from gstools import SRF, CovModel
import matplotlib.pyplot as plt
from matplotlib import colormaps
import gstools as gs

# from src.dataset.interfaces.idataset import IDataset
from src.dataset.spatial_dataset import SpatialDataset


class SimulatedSpatialDataset(SpatialDataset):

    field_size: int
    data_seed: int
    process_seed: list[int]
    len_scale_seed: list[int]
    error_seed: int
    k: int
    n: int
    coordinates: npt.NDArray[np.float64]

    def __init__(self,
                 field_size=40,
                 data_seed=222,
                 process_seed=[555, 888, 111],
                 len_scale_seed=[6, 12, 18],
                 error_seed=333,
                 k=2,
                 useIntercept=True
                 ) -> None:
        self.field_size = field_size
        self.data_seed = data_seed
        self.process_seed = process_seed
        self.len_scale_seed = len_scale_seed
        self.error_seed = error_seed
        self.k = k
        self.n = field_size * field_size
        self.useIntercept = useIntercept

    def generate_data(self):
        np.random.seed(self.data_seed)
        X_list = []
        for _ in range(self.k):
            X_list.append(
                np.random.randn(self.field_size * self.field_size)
            )
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

        self.X_original = self.X.copy()

        return [self.X]

    def generate_processes(self):

        class GWR_gau(CovModel):
            def correlation(self, r):
                return np.exp(-0.5 * (r / self.len_scale) ** 2)

        coords = [
            range(self.field_size),
            range(self.field_size)
        ]

        processes = []
        process_k = self.k
        if self.useIntercept:
            process_k = self.k + 1  # add one for intercept

        for i in range(process_k):
            
            if i == 1:

                len_smooth = 100
                model_smooth = gs.Gaussian(
                    dim=2,
                    var=1.0,
                    len_scale=len_smooth,
                )
                srf_smooth = SRF(
                    model_smooth,
                    mean=0.0,
                    seed=self.process_seed[i]
                )
                field_smooth_2d = srf_smooth.structured(coords).reshape(-1)

                # 3.2 Rough component (short length scale)
                len_rough = 5.0
                model_rough = gs.Gaussian(
                    dim=2,
                    var=1.0,
                    len_scale=len_rough,
                )
                srf_rough = gs.SRF(
                    model_rough,
                    mean=0.0,
                    seed=self.process_seed[i]
                )
                field_rough_2d = srf_rough.structured(coords).reshape(-1)


                x = np.arange(self.field_size)
                y = np.arange(self.field_size)

                # for making spatially varying weights later
                xx, yy = np.meshgrid(x, y, indexing="xy")
                w = (xx / (self.field_size - 1)) ** 2
                w = w.reshape(-1)

                process = field_smooth_2d + w * field_rough_2d
                process = (process - process.mean()) / process.std() + 2.0
                processes.append(process)
                continue

            model_L = GWR_gau(
                dim=2,
                var=1.0,
                len_scale=self.len_scale_seed[i]
            )
            srf_L = SRF(
                model_L,
                mean=0,
                seed=self.process_seed[i]
            )
            process = srf_L.structured(coords).reshape(-1)
            process = (process - process.mean()) / process.std() + 2.0
            processes.append(process)

        return [np.array(processes).T]

    def fit_y(self, X, beta):
        """
        Fit the response variable y based on the design matrix X and coefficients b0, b1, b2.
        """
        np.random.seed(self.error_seed)
        self.err = np.random.randn(self.field_size * self.field_size)
        self.y = np.sum(X * beta, axis=1) + self.err
        return [self.y, self.err]

    def plot(self, b, sub_title=['', '', '', ''], size=40,
            vmin=None, vmax=None, palette='viridis',
            points=None):
        """
        points: list or array of 1D indices to mark on the surface.
                Example: points=[idx1, idx2]
        """
        k = len(b)
        fig, axs = plt.subplots(1, k, figsize=(6*k, 4))
        if k == 1:
            axs = [axs]

        # convert points to (row, col)
        coords = []
        if points is not None:
            for p in points:
                r, c = divmod(int(p), size)
                coords.append((r, c))

        for i in range(k):
            ax_img = axs[i].imshow(
                b[i].reshape(size, size),
                cmap=colormaps[palette],
                vmin=vmin, vmax=vmax
            )

            axs[i].set_title(sub_title[i], fontsize=16)
            fig.colorbar(ax_img, ax=axs[i])

            # grid
            axs[i].set_xticks(np.arange(-0.5, size, 5))
            axs[i].set_yticks(np.arange(-0.5, size, 5))
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            axs[i].tick_params(axis='x', colors=(0, 0, 0, 0))
            axs[i].tick_params(axis='y', colors=(0, 0, 0, 0))

            # plot selected points
            if points is not None:
                for (rr, cc) in coords:
                    axs[i].scatter(cc, rr, c='red', s=60, marker='x')

        plt.show()
