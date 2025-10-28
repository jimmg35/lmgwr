import numpy as np

from mgwr.gwr import GWR as OfficialGWR
from mgwr.sel_bw import Sel_BW

from src.log.gwr_logger import GwrLogger
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.model.lgwr import LGWR
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.lgwr_logger import LgwrLogger
from src.optimizer.reinforce.lgwr_optimizer import LgwrOptimizerRL
from stable_baselines3 import PPO

# Hyperparameters for PPO training
MAX_STEPS = 1000
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -1.0
MAX_ACTION = 1.0
MIN_BANDWIDTH = 30

if __name__ == '__main__':

    logger = LgwrLogger()

    k = 1
    field_size = 40

    dataset_smooth = SimulatedSpatialDataset(
        k=k,
        field_size=field_size,
        process_seed=[666, 888],
        len_scale_seed=[100, 100]
    )
    [X_smooth] = dataset_smooth.generate_data()
    [beta_smooth] = dataset_smooth.generate_processes()
    [y_smooth, err_smooth] = dataset_smooth.fit_y(
        X_smooth, beta_smooth
    )


    # Create a LGWR kernel and LGWR model.
    kernel = LgwrKernel(
        dataset_smooth,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    lgwr = LGWR(dataset_smooth, kernel)

    # Initialize lgwr gym environment
    env = LgwrOptimizerRL(
        lgwr,
        logger,
        TOTAL_TIMESTEPS,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=dataset_smooth.X.shape[0],
        min_action=MIN_ACTION,
        max_action=MAX_ACTION,
        max_steps_per_episode=MAX_STEPS
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