
from stable_baselines3 import PPO
import numpy as np
import pandas as pd

from src.optimizer.reinforce.gwr_optimizer import GwrOptimizerRL
from src.dataset.interfaces.idataset import FieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.gwr_logger import GwrLogger
from src.model.gwr import GWR

# Hyperparameters for PPO training
MAX_STEPS = 500
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -10
MAX_ACTION = 10

MIN_BANDWIDTH = 30

# REWARD_TYPE = LgwrRewardType.AICC
# REWARD_THRESHOLD = 300


if __name__ == '__main__':

    # Create a logger to record the GWR model's information.
    logger = GwrLogger()

    # Create a simulated dataset.
    field_size = 40
    spatialDataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = spatialDataset.generate_processes()
    [X, y] = spatialDataset.fit_y(b0, b1, b2)

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(
        spatialDataset,
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
        max_bandwidth=spatialDataset.X.shape[0],
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
