
from stable_baselines3 import PPO
import pandas as pd
import numpy as np

from src.optimizer.reinforce.lgwr_optimizer import LgwrOptimizerRL, LgwrRewardType
from src.dataset.interfaces.idataset import FieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.lgwr_logger import LgwrLogger
from src.model.lgwr import LGWR

# Hyperparameters for PPO training
MAX_STEPS = 50000
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -10
MAX_ACTION = 10

MIN_BANDWIDTH = 30

REWARD_TYPE = LgwrRewardType.AICC
# REWARD_THRESHOLD = 300

if __name__ == '__main__':

    # Create a logger to record the LGWR model's information.
    logger = LgwrLogger()

    # Create a simulated dataset.
    field_size = 40
    spatialDataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = spatialDataset.generate_processes()
    [X, y] = spatialDataset.fit_y(b0, b1, b2)

    # Create a LGWR kernel and LGWR model.
    kernel = LgwrKernel(
        spatialDataset,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    lgwr = LGWR(spatialDataset, kernel, logger)

    # Initialize lgwr gym environment
    env = LgwrOptimizerRL(
        lgwr,
        logger,
        # REWARD_THRESHOLD,
        None,
        TOTAL_TIMESTEPS,
        reward_type=REWARD_TYPE,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=spatialDataset.X.shape[0],
        min_action=MIN_ACTION,
        max_action=MAX_ACTION,
        max_steps=MAX_STEPS
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
