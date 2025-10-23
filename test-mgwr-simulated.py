import numpy as np

from mgwr.gwr import MGWR as OfficialMGWR
from mgwr.sel_bw import Sel_BW

# from kernel.lgwr_kernel import LgwrKernel
from src.kernel.gwr_kernel import GwrKernel
from src.model.mgwr import MGWR
from src.model.gwr import GWR
from src.model.lmgwr import LMGWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset


from mgwr.gwr import GWR as OfficialGWR

k = 2
field_size = 40

dataset = SimulatedSpatialDataset(
    k=k,
    field_size=field_size
)
[X] = dataset.generate_data()
[beta] = dataset.generate_processes()
[y, err] = dataset.fit_y(X, beta)



kernel = GwrKernel(
    dataset,
    kernel_type='bisquare'
)
gwr = GWR(dataset, kernel)

for i in range(100, 1000, 100):
    gwr = gwr.update_bandwidth(i).fit()
    print(f"Bandwidth: {i}, AICc: {gwr.aicc}, tr(S): {gwr.tr_S}")
    # print(gwr.S[:10, :10])
    print(gwr.S[0, :])
    print(gwr.S[-1, :])

# g_X = dataset.X[:, 1:]
# g_y = dataset.y.reshape(-1, 1)
# g_coords = dataset.coordinates.tolist()
# gwr_selector = Sel_BW(g_coords, g_y, g_X)
# gwr_bw = gwr_selector.search(bw_min=2)
# official_gwr = OfficialGWR(g_coords, g_y, g_X, gwr_bw, hat_matrix=True)
# official_gwr_result = official_gwr.fit()
# print(gwr_bw)

# kernel = GwrKernel(dataset, 'bisquare')
# refactored_mgwr = MGWR(dataset, kernel)
# refactored_mgwr.update_bandwidth_set([46., 102., 93.]).exact_fit()


# lgwr_kernel = LgwrKernel(
#     dataset,
#     kernel_type='bisquare',
#     kernel_bandwidth_type='adaptive'
# )
# LMGWR_model = LMGWR(dataset, lgwr_kernel)
# LMGWR_model.update_bandwidth_matrix(np.array([[46., 102.], [93., 85.]])).exact_fit()


# # === OLS: hat matrix S, trace, preview ===
# X = dataset.X                       # n × p，需含截距
# y_vec = y.reshape(-1, 1)            # n × 1

# # β̂、Ŷ、殘差
# XtX_inv = np.linalg.inv(X.T @ X)    # (X'X)^{-1}
# beta_hat = XtX_inv @ (X.T @ y_vec)  # p × 1
# y_hat = X @ beta_hat                # n × 1
# e = y_vec - y_hat

# # OLS hat matrix
# S_ols = X @ XtX_inv @ X.T           # n × n
# tr_S_ols = np.trace(S_ols)          # 應等於 p（含截距）

# # 殘差自由度與σ̂²
# n, p = X.shape
# sigma2_hat = float((e.T @ e) / (n - p))

# print(f"OLS: n={n}, p={p}")
# print(f"trace(S_ols) = {tr_S_ols:.6f}  (should equal p)")
# print(f"Residual df = {n - p}, sigma^2 = {sigma2_hat:.6f}")

# # 預覽 S 的左上角 5×5（避免印出 1600×1600 全矩陣）
# np.set_printoptions(precision=4, suppress=True)
# print("S_ols[0:5, 0:5] =\n", S_ols[:5, :5])
# print(S_ols)