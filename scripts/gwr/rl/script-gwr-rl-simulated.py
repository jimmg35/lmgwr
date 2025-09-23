
import numpy as np
from stable_baselines3 import PPO

from src.optimizer.reinforce.gwr_optimizer import GwrOptimizerRL
from src.dataset.interfaces.idataset import FieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.log.gwr_logger import GwrLogger
from src.model.gwr import GWR
from src.utility.distribution_test import normal_distribution_test

# Hyperparameters for PPO training
MAX_STEPS = 10
TOTAL_TIMESTEPS = MAX_STEPS * 10
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
    dataset = SimulatedSpatialDataset(field_size=field_size)
    [b0, b1, b2] = dataset.generate_processes()
    [X, y, err] = dataset.fit_y(b0, b1, b2)

    # Create a GWR kernel and GWR model.
    kernel = GwrKernel(
        dataset,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    gwr = GWR(dataset, kernel, logger)

    # Initial gwr gym environment
    env = GwrOptimizerRL(
        gwr,
        logger,
        # REWARD_THRESHOLD,
        None,
        TOTAL_TIMESTEPS,
        min_bandwidth=MIN_BANDWIDTH,
        max_bandwidth=dataset.X.shape[0],
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

    print(gwr.betas.shape)
    [b0_gwr, b1_gwr, b2_gwr] = [gwr.betas[:, 0], gwr.betas[:, 1], gwr.betas[:, 2]]
    # dataset.plot(
    #     b=np.vstack([b0_gwr, b1_gwr, b2_gwr]),
    #     sub_title=['b0_gwr', 'b1_gwr', 'b2_gwr'],
    #     size=field_size
    # )

    pred_y = (b0_gwr * dataset.X[:, 0] + b1_gwr * dataset.X[:, 1] + b2_gwr *
                dataset.X[:, 2] + dataset.err).reshape(-1, 1)
    pred_err = pred_y - y
    # dataset.plot(
    #     b=np.vstack(pred_err.T),
    #     sub_title=['prediction error'],
    #     size=field_size
    # )
    normal_distribution_test(pred_err, title='Prediction Error', xlabel='Error')

