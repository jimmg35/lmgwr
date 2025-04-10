import numpy as np
import pandas as pd
import geopandas as gp

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import FieldInfo


def test_spatial_dataset():

    georgia_data = pd.read_csv(r'./data/GData_utm.csv')
    georgia_shp = gp.read_file(r'./data/G_utm.shp')

    spatialDataset = SpatialDataset(
        georgia_data,
        FieldInfo(
            predictor_fields=['PctBach', 'PctEld', 'TotPop90'],
            response_field='PctPov',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        geometry=georgia_shp,
        isSpherical=False,
        useIntercept=True,
        isStandardize=True
    )

    # Check if the dataset is set up correctly
    assert spatialDataset.fieldInfo is not None
    assert spatialDataset.geometry is not None
    assert spatialDataset.logger is None
    assert spatialDataset.isSpherical is False
    assert spatialDataset.useIntercept is True
    assert spatialDataset.isStandardize is True
    assert len(spatialDataset) == 159
    assert isinstance(spatialDataset.X, np.ndarray)
    assert isinstance(spatialDataset.y, np.ndarray)
    assert isinstance(spatialDataset.coordinates, np.ndarray)
