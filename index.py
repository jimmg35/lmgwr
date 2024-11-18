import numpy as np
import pandas as pd

from src.log.logger import create_logger
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel

from scipy import linalg

create_logger()


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/GData_utm.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'PctBlack'],
            response_field='PctPov',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        isSpherical=True
    )

    kernel = GwrKernel(spatialDataset, 300, 'bisquare')
    gwr = GWR(spatialDataset, kernel)
    gwr.fit()

    # # Initializing the estimates storing variables.
    # import numpy.typing as npt
    # betas: npt.NDArray[np.float64] = np.empty(
    #     (20, 1), dtype=np.float64)

    # # Ensure the initialized values are null.
    # betas.fill(np.nan)

    # print(betas)
    # print(betas[0])

    # betas[0] = 5.51
    # print(betas)
    # print(betas[0])
