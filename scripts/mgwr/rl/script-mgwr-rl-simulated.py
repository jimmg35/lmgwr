from stable_baselines3 import PPO

from src.optimizer.reinforce.mgwr_optimizer import MgwrOptimizerRL
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.mgwr_logger import MgwrLogger
from src.model.mgwr import MGWR

# Hyperparameters for PPO training
MAX_STEPS = 1000
TOTAL_TIMESTEPS = MAX_STEPS * 1000
MIN_ACTION = -1.0
MAX_ACTION = 1.0
MIN_BANDWIDTH = 30


if __name__ == '__main__':

    # Create a logger to record the MGWR model's information.
    logger = MgwrLogger()

    # Create a simulated dataset.
    field_size = 40
    spatialDataset = SimulatedSpatialDataset(field_size=field_size, k=3)
    [X] = spatialDataset.generate_data()
    [beta] = spatialDataset.generate_processes()
    [y, err] = spatialDataset.fit_y(X, beta)

    # Create an MGWR kernel and MGWR model.
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
        max_action=MAX_ACTION,
        max_steps_per_episode=MAX_STEPS
    )

    # Using PPO to optimize the bandwidth set
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

