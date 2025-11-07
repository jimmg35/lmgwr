from src.reload.lgwr_reload import reload_logs, reload_lgwr
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel

# Create synthetic dataset
field_size = 40
dataset = SimulatedSpatialDataset(field_size=field_size)
[X] = dataset.generate_data()
[beta] = dataset.generate_processes()
[y, err] = dataset.fit_y(X, beta)

# Reload bandwidth sets from log file
bandwidth_sets = reload_logs(r"./logs/lgwr-2025-10-23-23-07-47-log/model_info.json")





# model = reload_lgwr(dataset, bandwidth_sets[0])



# dataset.plot(
#     b=beta.T,
#     sub_title=[r"True $\beta_0$", r"True $\beta_1$", r"True $\beta_2$"],
#     size=field_size,
#     vmin=beta.min(),
#     vmax=beta.max()
# )


# dataset.plot(
#     b=model.betas.T,
#     sub_title=[r"Estimated $\beta_0$", r"Estimated $\beta_1$", r"Estimated $\beta_2$"],
#     size=field_size,
#     vmin=beta.min(),
#     vmax=beta.max()
# )