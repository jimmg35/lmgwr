import os
import json
import pandas as pd
import geopandas as gp

from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.visualize.lgwr_visualizer import LgwrVisualizer
from src.log.lgwr_logger import LgwrLogger


if __name__ == "__main__":

    log_dir = r'./locked-logs/georgia/03-super-large-episode'
    lgwr_log_files = os.listdir(log_dir)

    logger = LgwrLogger()
    georgia_data = pd.read_csv(r'./data/GData_utm.csv')
    georgia_shp = gp.read_file(r'./data/G_utm.shp')
    spatialDataset = SpatialDataset(
        georgia_data,
        IFieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        logger,
        isSpherical=True,
        geometry=georgia_shp
    )

    for log_path in lgwr_log_files:

        lgwr_visualizer = LgwrVisualizer(
            os.path.join(log_dir, log_path),
            spatialDataset
        )

        with open(
            os.path.join(os.path.join(log_dir, log_path),
                         'model_info.json'), 'r'
        ) as file:
            model_info = json.load(file)

            for episode_record in model_info["bandwidth_optimization"]:

                episode = episode_record['episode']
                result = lgwr_visualizer.descriptive_statistics(episode)
                lgwr_visualizer.plot_bandwidth_map_by_episode(
                    episode, result, save=True
                )
                lgwr_visualizer.plot_training_process(
                    episode
                )
                print(result)
