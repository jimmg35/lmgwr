from stable_baselines3 import PPO

from src.optimizer.reinforce.lmgwr_optimizer import LmgwrOptimizerRL
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.lgwr_kernel import LgwrKernel
from src.log.lmgwr_logger import LmgwrLogger
from src.model.lmgwr import LMGWR

# Hyperparameters for PPO training
MAX_STEPS = 1000
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -1.0
MAX_ACTION = 1.0
MIN_BANDWIDTH = 30


if __name__ == '__main__':

    # Create a logger to record the LMGWR model's information.
    logger = LmgwrLogger()

    # Create a simulated dataset.
    field_size = 40
    spatialDataset = SimulatedSpatialDataset(field_size=field_size)
    [X] = spatialDataset.generate_data()
    [beta] = spatialDataset.generate_processes()
    [y, err] = spatialDataset.fit_y(X, beta)

    # Create a LMGWR kernel and LMGWR model.
    kernel = LgwrKernel(
        spatialDataset,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    lmgwr = LMGWR(spatialDataset, kernel)

    # Initialize LMGWR gym environment
    env = LmgwrOptimizerRL(
        lmgwr,
        logger,
        TOTAL_TIMESTEPS,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=spatialDataset.X.shape[0],
        min_action=MIN_ACTION,
        max_action=MAX_ACTION,
        max_steps_per_episode=MAX_STEPS
    )

    # Using PPO to optimize the bandwidth matrix
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

