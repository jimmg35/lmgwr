import numpy as np
import numpy.typing as npt
from gstools import SRF, CovModel

from src.dataset.interfaces.idataset import IDataset


class SimulatedSpatialDataset(IDataset):

    field_size: int
    data_seed: int
    k: int
    coordinates: npt.NDArray[np.float64]

    def __init__(self,
                 field_size=40,
                 data_seed=222,
                 process_seed=[555, 888],
                 k=2
                 ) -> None:
        self.field_size = field_size
        self.data_seed = data_seed
        self.k = k

        self.__generate_data()

    def __generate_data(self):
        np.random.seed(self.data_seed)
        X_list = []
        for _ in range(self.k):
            X_list.append(np.random.randn(self.field_size * self.field_size))
        u = np.array([np.linspace(0, self.field_size-1,
                     num=self.field_size)]*self.field_size).reshape(-1)
        v = np.array([np.linspace(0, self.field_size-1,
                     num=self.field_size)]*self.field_size).T.reshape(-1)
        self.X = np.vstack(X_list)
        self.coordinates = np.array(list(zip(u, v)))

    # def generate_processes(self):

    #     class GWR_gau(CovModel):
    #         def correlation(self, r):
    #             return np.exp(-0.5 * (r / self.len_scale) ** 2)

    #     model_b0 = GWR_gau(dim=2, var=1, len_scale=6)
    #     model_b1 = GWR_gau(dim=2, var=1, len_scale=12)

    #     srf_b0 = SRF(model_b0, mean=0, seed=seed_b0)
    #     srf_b1 = SRF(model_b1, mean=0, seed=seed_b1)

    #     coords = [range(field_size), range(field_size)]

    #     b0 = srf_b0.structured(coords).reshape(-1)
    #     b1 = srf_b1.structured(coords).reshape(-1)

    #     # Normalize
    #     b0 = (b0 - b0.mean()) / b0.std() + 2
    #     b1 = (b1 - b1.mean()) / b1.std() + 2
    #     b2 = np.ones(field_size*field_size).reshape(-1)*2

    #     return b0, b1, b2
