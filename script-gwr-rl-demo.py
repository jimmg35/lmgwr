
from stable_baselines3 import PPO
import pandas as pd

from src.optimizer.reinforce.gwr_optimizer import GwrOptimizerRL
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.gwr_logger import GwrLogger
from src.model.gwr import GWR

# Hyperparameters for PPO training
MAX_STEPS = 50000
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -10
MAX_ACTION = 10

MIN_BANDWIDTH = 30

# REWARD_TYPE = LgwrRewardType.AICC
# REWARD_THRESHOLD = 300


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

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

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(
        spatialDataset,
        logger,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    gwr = GWR(spatialDataset, kernel, logger)

    # Initial gwr gym environment
    env = GwrOptimizerRL(
        gwr,
        logger,
        # REWARD_THRESHOLD,
        None,
        TOTAL_TIMESTEPS,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=spatialDataset.x_matrix.shape[0],
        min_action=MIN_ACTION,
        max_action=MAX_ACTION,
        max_steps=MAX_STEPS
    )

    # Using PPO to optimize the bandwidth
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
