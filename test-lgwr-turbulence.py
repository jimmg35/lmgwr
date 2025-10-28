
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

    dataset_turbulence = SimulatedSpatialDataset(
        k=k,
        field_size=field_size,
        process_seed=[666, 888],
        len_scale_seed=[100, 2]
    )
    [X_turbulence] = dataset_turbulence.generate_data()
    [beta_turbulence] = dataset_turbulence.generate_processes()
    [y_turbulence, err_turbulence] = dataset_turbulence.fit_y(
        X_turbulence, beta_turbulence
    )


    # Create a LGWR kernel and LGWR model.
    kernel = LgwrKernel(
        dataset_turbulence,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    lgwr = LGWR(dataset_turbulence, kernel)

    # Initialize lgwr gym environment
    env = LgwrOptimizerRL(
        lgwr,
        logger,
        TOTAL_TIMESTEPS,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=dataset_turbulence.X.shape[0],
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