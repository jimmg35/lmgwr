
from stable_baselines3 import PPO
import pandas as pd

from src.optimizer.reinforce.mgwr_optimizer import MgwrOptimizerRL
from src.dataset.interfaces.idataset import FieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.mgwr_logger import MgwrLogger
from src.model.mgwr import MGWR

# Hyperparameters for PPO training
MAX_STEPS = 50000
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -10
MAX_ACTION = 10

MIN_BANDWIDTH = 30

if __name__ == '__main__':

    # Create a logger to record the MGWR model's information.
    logger = MgwrLogger()

    # Load the Georgia dataset and create a spatial dataset.
    georgia_data = pd.read_csv(r'./data/GData_utm.csv')
    spatialDataset = SpatialDataset(
        georgia_data,
        FieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='Longitud',
            coordinate_y_field='Latitude'
        ),
        isSpherical=True
    )

    # Create a MGWR kernel and MGWR model.
    kernel = GwrKernel(
        spatialDataset,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    mgwr = MGWR(spatialDataset, kernel)

    # Initialize MGWR gym environment
    env = MgwrOptimizerRL(
        mgwr,
        logger,
        TOTAL_TIMESTEPS,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=spatialDataset.X.shape[0],
        min_action=MIN_ACTION,
        max_action=MAX_ACTION
    )

    # Using PPO to optimize the bandwidth vector
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device='cpu'
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS
    )
    logger.append_info("PPO: PPO finished training.")
