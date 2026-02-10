import numpy as np
import numpy.typing as npt
from gstools import SRF, CovModel
import matplotlib.pyplot as plt
from matplotlib import colormaps
import gstools as gs
import plotly.graph_objects as go

from .dgp.random_locations_controlled import process2_composite_random_blobs


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
                 process_seed=[888],
                 len_scale_seed=[1, 0.1],
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

        processes = []
        process_k = self.k + 1 if self.useIntercept else self.k

        for i in range(process_k):

            # DGP settings
            length_scale_base = 1   
            length_scale_mutate = 0.1
            sigma_process = 1.0
            seed_eps = 888
            seed_blobs = 0
            composite_n_patches = 12
            composite_r_min_cells = 3
            composite_r_max_cells = 10

            # Process 2: composite
            X2, Y2, Beta_composite, Z_short, Z_long, W, coords2, eps2 = process2_composite_random_blobs(
                n=self.field_size,
                length_scale_short=length_scale_mutate,
                length_scale_long=length_scale_base,
                sigma=sigma_process,
                n_patches=composite_n_patches,
                r_min_cells=composite_r_min_cells,
                r_max_cells=composite_r_max_cells,
                seed_eps=seed_eps,      # <-- SAME ε as Process 1
                seed_blobs=seed_blobs
            )
            

            # process = (process - process.mean()) / process.std() + 2.0
            processes.append(Beta_composite.reshape(-1, 1))

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

    def plot_process_3d(
        self,
        process: npt.ArrayLike,
        title: str = "Process surface",
        show: bool = True,
        save_html: str | None = None,
        colorscale: str = "Viridis",
        opacity: float = 0.95,
        points: list[int] | None = None,
        point_labels: list[str] | None = None,
    ) -> go.Figure:
        """
        Plot a 3D interactive surface for a process/beta field.

        Parameters
        ----------
        process:
            Array of shape (n,) or (field_size, field_size).
        title:
            Figure title.
        show:
            If True, call fig.show().
        save_html:
            If not None, write to this html path (e.g., "surface.html").
        points:
            Optional list of 1D indices (0..n-1) to mark on the surface.
        point_labels:
            Optional labels for points (same length as points).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if not hasattr(self, "field_size"):
            raise AttributeError("Dataset must have `field_size` attribute.")
        if not hasattr(self, "coordinates"):
            raise AttributeError("Dataset must have `coordinates` generated (call generate_data first).")

        fs = int(self.field_size)
        n_expected = fs * fs

        Z = np.asarray(process)
        if Z.ndim == 1:
            if Z.size != n_expected:
                raise ValueError(f"process has length {Z.size}, expected {n_expected} (= field_size^2).")
            Z = Z.reshape(fs, fs)
        elif Z.ndim == 2:
            if Z.shape != (fs, fs):
                raise ValueError(f"process has shape {Z.shape}, expected {(fs, fs)}.")
        else:
            raise ValueError("process must be 1D or 2D array.")

        # Build X,Y grid from your coordinates convention:
        # In your SimulatedSpatialDataset, coordinates are (u, v) with u as x, v as y on a fs-by-fs grid.
        coords = np.asarray(self.coordinates)
        if coords.shape != (n_expected, 2):
            raise ValueError(f"coordinates has shape {coords.shape}, expected {(n_expected, 2)}")

        xs = np.unique(coords[:, 0])
        ys = np.unique(coords[:, 1])
        if xs.size != fs or ys.size != fs:
            # fallback: assume 0..fs-1
            xs = np.arange(fs)
            ys = np.arange(fs)

        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=Xg,
            y=Yg,
            z=Z,
            colorscale=colorscale,
            opacity=opacity,
            showscale=True
        ))

        # Optional: mark selected points
        if points is not None and len(points) > 0:
            pts = [int(p) for p in points]
            rows = [p // fs for p in pts]
            cols = [p % fs for p in pts]

            px = Xg[rows, cols]
            py = Yg[rows, cols]
            pz = Z[rows, cols]

            if point_labels is None:
                point_labels = [f"idx={p}" for p in pts]
            if len(point_labels) != len(pts):
                raise ValueError("point_labels must have same length as points.")

            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers+text",
                marker=dict(size=6, color="red"),
                text=point_labels,
                textposition="top center",
                name="points"
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="value",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if save_html is not None:
            fig.write_html(save_html)

        if show:
            fig.show()

        return fig
