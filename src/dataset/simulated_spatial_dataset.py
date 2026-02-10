import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colormaps
import plotly.graph_objects as go

from src.dataset.spatial_dataset import SpatialDataset
from .dgp.random_locations_controlled import process2_composite_random_blobs


class SimulatedSpatialDataset(SpatialDataset):
    """
    Simulated spatial dataset with composite-scale coefficient processes.

    Data-generating structure:
        X      ~ i.i.d. N(0, 1)
        beta(s) = composite spatial process
        y(s)   = X(s) * beta(s) + error(s)
        error  ~ i.i.d. N(0, sigma_error^2)
    """

    field_size: int
    data_seed: int
    process_seed: list[int]
    len_scale_seed: list[float]
    error_seed: int
    k: int
    n: int
    coordinates: npt.NDArray[np.float64]

    def __init__(
        self,
        field_size: int = 40,
        data_seed: int = 456,
        process_seed: list[int] = [888],
        len_scale_seed: list[float] = [1.0, 0.1],
        error_seed: int = 789,
        k: int = 1,
        useIntercept: bool = False,
    ) -> None:
        self.field_size = field_size
        self.data_seed = data_seed
        self.process_seed = process_seed
        self.len_scale_seed = len_scale_seed
        self.error_seed = error_seed
        self.k = k
        self.useIntercept = useIntercept
        self.n = field_size * field_size

    # ------------------------------------------------------------------
    # 1. Generate covariates and spatial coordinates
    # ------------------------------------------------------------------

    def generate_data(self):
        """
        Generate covariates X and spatial coordinates.

        Returns
        -------
        list containing X : ndarray (n, p)
        """

        rng = np.random.default_rng(self.data_seed)

        X_list = []
        for _ in range(self.k):
            X_list.append(rng.normal(0.0, 1.0, size=self.n))

        self.X = np.vstack(X_list).T  # (n, k)

        if self.useIntercept:
            self.X = np.hstack([np.ones((self.n, 1)), self.X])

        # Regular grid coordinates
        xs = np.linspace(0, self.field_size - 1, self.field_size)
        ys = np.linspace(0, self.field_size - 1, self.field_size)
        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

        self.coordinates = np.column_stack([Xg.ravel(), Yg.ravel()])
        self.X_original = self.X.copy()

        return [self.X]

    # ------------------------------------------------------------------
    # 2. Generate spatial coefficient processes (true beta fields)
    # ------------------------------------------------------------------

    def generate_processes(self):
        """
        Generate spatially varying coefficient surfaces using
        a composite-scale Gaussian random field.

        Returns
        -------
        list containing beta : ndarray (n, p)
        """

        processes = []
        process_k = self.k + 1 if self.useIntercept else self.k

        for i in range(process_k):

            # ---- Composite DGP settings ----
            n_grid = 60
            length_scale_base = 1   
            length_scale_mutate = 0.1
            sigma_process = 1.0
            seed_eps = 888
            seed_blobs = 0
            composite_n_patches = 12
            composite_r_min_cells = 3
            composite_r_max_cells = 10
            seed_x = 456
            seed_err = 789
            sigma_error = 0.5

            _, _, beta_comp, _, _, _, _, _ = process2_composite_random_blobs(
                n=self.field_size,
                length_scale_short=length_scale_mutate,
                length_scale_long=length_scale_base,
                sigma=sigma_process,
                n_patches=composite_n_patches,
                r_min_cells=composite_r_min_cells,
                r_max_cells=composite_r_max_cells,
                seed_eps=seed_eps,
                seed_blobs=seed_blobs,
            )

            beta_vec = beta_comp.reshape(-1)

            # Optional standardization for numerical stability
            # beta_vec = (beta_vec - beta_vec.mean()) / beta_vec.std()

            processes.append(beta_vec)

        beta = np.vstack(processes).T  # (n, p)
        return [beta]

    # ------------------------------------------------------------------
    # 3. Generate response variable
    # ------------------------------------------------------------------

    def fit_y(self, X, beta, sigma_error: float = 0.5):
        """
        Generate response variable y according to:
            y = X beta + error

        Parameters
        ----------
        X : ndarray (n, p)
            Design matrix.
        beta : ndarray (n, p)
            Spatially varying coefficients.
        sigma_error : float
            Standard deviation of Gaussian noise.

        Returns
        -------
        list containing:
            y   : ndarray (n,)
            err : ndarray (n,)
        """

        rng = np.random.default_rng(self.error_seed)
        self.err = rng.normal(0.0, sigma_error, size=X.shape[0]).reshape(-1, 1)

        beta2_vec = beta.reshape(self.field_size * self.field_size, 1)
        self.y = beta2_vec * X + self.err

        return [self.y, self.err]

    # ------------------------------------------------------------------
    # 4. 2D plotting utilities
    # ------------------------------------------------------------------

    def plot(
        self,
        b,
        sub_title=None,
        size=None,
        vmin=None,
        vmax=None,
        palette="viridis",
        points=None,
    ):
        if size is None:
            size = self.field_size
        if sub_title is None:
            sub_title = [""] * len(b)

        k = len(b)
        fig, axs = plt.subplots(1, k, figsize=(6 * k, 4))
        if k == 1:
            axs = [axs]

        coords_pts = []
        if points is not None:
            for p in points:
                r, c = divmod(int(p), size)
                coords_pts.append((r, c))

        for i in range(k):
            ax_img = axs[i].imshow(
                b[i].reshape(size, size),
                cmap=colormaps[palette],
                vmin=vmin,
                vmax=vmax,
            )
            axs[i].set_title(sub_title[i], fontsize=14)
            fig.colorbar(ax_img, ax=axs[i])

            if points is not None:
                for (rr, cc) in coords_pts:
                    axs[i].scatter(cc, rr, c="red", s=60, marker="x")

        plt.show()

    # ------------------------------------------------------------------
    # 5. Interactive 3D surface plot
    # ------------------------------------------------------------------

    def plot_process_3d(
        self,
        process,
        title="Process surface",
        show=True,
        save_html=None,
        colorscale="Viridis",
        opacity=0.95,
    ) -> go.Figure:

        fs = self.field_size
        Z = np.asarray(process).reshape(fs, fs)

        xs = np.arange(fs)
        ys = np.arange(fs)
        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

        fig = go.Figure(
            data=[
                go.Surface(
                    x=Xg,
                    y=Yg,
                    z=Z,
                    colorscale=colorscale,
                    opacity=opacity,
                    showscale=True,
                )
            ]
        )

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
