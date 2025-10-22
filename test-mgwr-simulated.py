import numpy as np

from mgwr.gwr import MGWR as OfficialMGWR
from mgwr.sel_bw import Sel_BW

from kernel.lgwr_kernel import LgwrKernel
from src.kernel.gwr_kernel import GwrKernel
from src.model.mgwr import MGWR
from src.model.lmgwr import LMGWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset

k = 2
field_size = 40

dataset = SimulatedSpatialDataset(
    k=k,
    field_size=field_size
)
[X] = dataset.generate_data()
[beta] = dataset.generate_processes()
[y, err] = dataset.fit_y(X, beta)

# dataset.plot(
#     b=beta.T,
#     sub_title=[r"True $\beta_0$", r"True $\beta_1$", r"True $\beta_2$"],
#     size=field_size
# )

# dataset.plot(
#     b=X.T,
#     sub_title=[r"Intercept", r"$X_2$", r"$X_3$"],
#     size=field_size
# )

# dataset.plot(
#     b=y.reshape(1, -1),
#     sub_title=['y'],
#     size=field_size
# )

# dataset.plot(
#     b=err.reshape(1, -1),
#     sub_title=['error term'],
#     size=field_size
# )

kernel = GwrKernel(dataset, 'bisquare')
refactored_mgwr = MGWR(dataset, kernel)
refactored_mgwr.update_bandwidth_set([46., 102., 93.]).exact_fit()


lgwr_kernel = LgwrKernel(
    dataset,
    kernel_type='bisquare',
    kernel_bandwidth_type='adaptive'
)
LMGWR_model = LMGWR(dataset, lgwr_kernel)
LMGWR_model.update_bandwidth_matrix(np.array([[46., 102.], [93., 85.]])).exact_fit()