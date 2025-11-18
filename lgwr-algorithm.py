import os
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
    一次「掃過所有點」的 local bandwidth 搜尋：
      - 若給 init_vec，就以該向量作為 base_vec
      - 否則以 init_global_bw 填滿一個向量作為 base_vec
    每個點 i 的搜尋邏輯：
      - 固定 base_vec 其他位置，僅替換第 i 個元素為候選帶寬
      - 以 AICc 最小的帶寬當作該點的最佳值
    """

    if candidate_bws is None:
        candidate_bws = np.arange(50, 201, 5)
    candidate_bws = np.asarray(candidate_bws, dtype=int)

    # 決定 base_vec
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

    # 先用 base_vec 跑一次 LGWR
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


def run_iterative_bandwidth_optimization(
    lgwr,
    dataset,
    candidate_bws=None,
    global_optimal_bw: float | int | Any | None = None,
    n_iter: int = 1,
    init_file: str | None = None,
    output_prefix: str = "final_local_bandwidths",
    start_index: int = 1,
):
    """
    多輪疊代版本：
      - 若 init_file 為 None：以 global_optimal_bw 填滿向量當初始值
      - 若 init_file 給定：讀入該 txt 的帶寬向量作為初始值
      - 每一輪跑完後，輸出 final_local_bandwidths_i.txt
        其中 i = start_index, start_index+1, ...

    回傳：最後一輪的帶寬向量
    """

    if candidate_bws is None:
        candidate_bws = np.arange(50, 201, 5)

    n_points = dataset.n

    # 1. 決定初始向量
    if init_file is not None:
        if not os.path.exists(init_file):
            raise FileNotFoundError(f"init_file not found: {init_file}")
        print(f"Loading initial bandwidths from {init_file}")
        current_vec = np.loadtxt(init_file, dtype=int)
        if current_vec.shape[0] != n_points:
            raise ValueError(
                f"Bandwidth vector in {init_file} has length {current_vec.shape[0]}, "
                f"expected {n_points}."
            )
    else:
        if global_optimal_bw is None:
            raise ValueError(
                "global_optimal_bw must be provided when init_file is None."
            )
        print("Initializing with global optimal bandwidth.")
        current_vec = np.full(n_points, int(global_optimal_bw), dtype=int)

    # 2. 疊代優化
    for it in range(n_iter):
        iter_idx = start_index + it
        print(f"\n==============================")
        print(f" Iteration {iter_idx}")
        print(f"==============================")

        current_vec = pick_local_bandwidths_exhaustive(
            lgwr=lgwr,
            n_points=n_points,
            candidate_bws=candidate_bws,
            init_vec=current_vec,          # 關鍵：用上一輪結果當 base_vec
            init_global_bw=None,           # 此輪不用 global_bw
        )

        out_name = f"{output_prefix}_{iter_idx}.txt"
        np.savetxt(out_name, current_vec, fmt="%d")
        print(f"Saved bandwidths to {out_name}")

    return current_vec


########################################
#  使用方式示例
########################################

# 情境 1：沒有任何 txt，從 global_optimal_bandwidth 開始疊代 10 輪
# 最後會產生 final_local_bandwidths_1.txt, ..., final_local_bandwidths_10.txt
# final_vec = run_iterative_bandwidth_optimization(
#     lgwr=lgwr,
#     dataset=dataset,
#     candidate_bws=np.arange(50, 201, 5),
#     global_optimal_bw=global_optimal_bandwidth,
#     n_iter=10,
#     init_file=None,
#     output_prefix="final_local_bandwidths",
#     start_index=1,
# )

# 情境 2：你已經有第一次跑出的 final_local_bandwidths.txt，
#          想接著從那個向量開始再疊代 1000 輪
# 會產生 final_local_bandwidths_1.txt ... final_local_bandwidths_1000.txt
#（初始的 final_local_bandwidths.txt 不會被覆寫）
final_vec = run_iterative_bandwidth_optimization(
    lgwr=lgwr,
    dataset=dataset,
    candidate_bws=np.arange(50, 201, 5),
    global_optimal_bw=None,  # 已經有 init_file，不需要 global_optimal_bw
    n_iter=1000,
    init_file="final_local_bandwidths.txt",
    output_prefix="final_local_bandwidths",
    start_index=1,
)
