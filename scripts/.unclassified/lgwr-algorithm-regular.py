import numpy as np
from mgwr.gwr import GWR as OfficialGWR
from mgwr.sel_bw import Sel_BW
from pyparsing import Any
from src.log.gwr_logger import GwrLogger
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.model.lgwr import LGWR
from src.kernel.lgwr_kernel import LgwrKernel

## Define the simulated dataset
k = 1
field_size = 40
dataset = SimulatedSpatialDataset(
    k=k,
    field_size=field_size,
    process_seed=[666, 888],
    len_scale_seed=[100, 10]
)
[X] = dataset.generate_data()
[beta] = dataset.generate_processes()
[y, err] = dataset.fit_y(
    X, beta
)

dataset.plot(
    b=np.vstack([
        beta.T
    ]),
    sub_title=[
        r"True $\beta_0$",
        r"True $\beta_1$"
    ],
    size=field_size
)

## Find the global optimal bandwidth
g_X = dataset.X[:, 1:]
g_y = dataset.y.reshape(-1, 1)
g_coords = dataset.coordinates.tolist()
gwr_selector = Sel_BW(
    g_coords,
    g_y,
    g_X
)
global_optimal_bandwidth = gwr_selector.search(bw_min=2)

###############################
#                             #
#  Initialize the LGWR model  #
#                             #
###############################
kernel = LgwrKernel(
    dataset,
    kernel_type='bisquare',
    kernel_bandwidth_type='adaptive'
)
lgwr = LGWR(dataset, kernel)


def pick_local_bandwidths_exhaustive(
    lgwr,
    n_points: int,
    candidate_bws=None,
    init_vec: np.ndarray | None = None,
    init_global_bw: float | int | Any | None = None,
):
    """
    一次「�??��??��??��? local bandwidth ?��?�?
      - ?�給 init_vec，就以該?��?作為 base_vec
      - ?��?�?init_global_bw 填滿一?��??��???base_vec
    每個�? i ?��?尋�?輯�?
      - ?��? base_vec ?��?位置，�??��?�?i ?��?素為?�選帶寬
      - �?AICc ?�小�?帶寬�??��?該�??��?佳�?
    """

    if candidate_bws is None:
        candidate_bws = np.arange(50, 201, 5)
    candidate_bws = np.asarray(candidate_bws, dtype=int)

    # 決�? base_vec
    if init_vec is not None:
        base_vec = np.asarray(init_vec, dtype=int).copy()
        if base_vec.shape[0] != n_points:
            raise ValueError(
                f"init_vec length ({base_vec.shape[0]}) "
                f"does not match n_points ({n_points})."
            )
    elif init_global_bw is not None:
        base_vec = np.full(n_points, int(init_global_bw), dtype=int)
    else:
        raise ValueError("Either init_vec or init_global_bw must be provided.")

    # ?�用 base_vec 跑�?�?LGWR
    lgwr.update_local_bandwidth_vector(base_vec)
    lgwr.fit()

    best_bw = base_vec.copy()

    for i in range(n_points):
        print(f"Optimizing bandwidth for point {i+1}/{n_points}")
        aicc_min = np.inf
        bw_min = base_vec[i]

        for j, bw in enumerate(candidate_bws):
            print(f" - testing candidate bandwidth {bw} ({j+1}/{len(candidate_bws)})")
            vec = base_vec.copy()
            vec[i] = int(bw)

            lgwr.update_local_bandwidth_vector(vec)
            lgwr.fit()
            aicc = float(lgwr.aicc)

            if aicc < aicc_min:
                aicc_min = aicc
                bw_min = bw

        best_bw[i] = bw_min

    return best_bw


########################################
#  使用?��?示�?
########################################

# 單次 pick_local_bandwidths_exhaustive，初始化用 global_optimal_bandwidth
candidate_bws = np.arange(50, 201, 5)
final_vec = pick_local_bandwidths_exhaustive(
    lgwr=lgwr,
    n_points=dataset.n,
    candidate_bws=candidate_bws,
    init_vec=None,
    init_global_bw=global_optimal_bandwidth,
)
np.savetxt("final_local_bandwidths.txt", final_vec, fmt="%d")
print("Saved final_local_bandwidths.txt")
