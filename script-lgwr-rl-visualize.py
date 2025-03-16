import pandas as pd

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.visualize.lgwr_visualizer import LgwrVisualizer
from src.log.lgwr_logger import LgwrLogger


if __name__ == "__main__":
    log_path = r'./logs/lgwr-2025-03-16-10-30-49-log'

    logger = LgwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    georgia_data = pd.read_csv(r'./data/GData_utm.csv')
    spatialDataset = SpatialDataset(
        georgia_data,
        IFieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        logger,
        isSpherical=True
    )

    spatialDataset.plot_map()

    # lgwr_visualizer = LgwrVisualizer(log_path)
    # lgwr_visualizer.plot_bandwidth_map_by_episode(4)

    # lgwr_visualizer.summary_bandwidth_info()
