import numpy as np
from gstools import SRF, CovModel

from dataset.interfaces.idataset import IDataset


class SimulatedSpatialDataset(IDataset):

    size: int
    seed: int
    k: int

    def generate_data(self, seed=222, size=40):
        np.random.seed(seed)
        X1 = np.random.randn(size * size)
        X2 = np.random.randn(size * size)
        u = np.array([np.linspace(0, size-1, num=size)]*size).reshape(-1)
        v = np.array([np.linspace(0, size-1, num=size)]*size).T.reshape(-1)
        coordinates = list(zip(u, v))
        return X1, X2, coordinates

    def generate_processes(self, size=40, seed_b0=555, seed_b1=888):

        class GWR_gau(CovModel):
            def correlation(self, r):
                return np.exp(-0.5 * (r / self.len_scale) ** 2)

        model_b0 = GWR_gau(dim=2, var=1, len_scale=6)
        model_b1 = GWR_gau(dim=2, var=1, len_scale=12)

        srf_b0 = SRF(model_b0, mean=0, seed=seed_b0)
        srf_b1 = SRF(model_b1, mean=0, seed=seed_b1)

        coords = [range(size), range(size)]

        b0 = srf_b0.structured(coords).reshape(-1)
        b1 = srf_b1.structured(coords).reshape(-1)

        # Normalize
        b0 = (b0 - b0.mean()) / b0.std() + 2
        b1 = (b1 - b1.mean()) / b1.std() + 2
        b2 = np.ones(size*size).reshape(-1)*2

        return b0, b1, b2
