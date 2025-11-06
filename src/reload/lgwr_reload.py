
from pathlib import Path
import json
import ast
import numpy as np

from src.model.lgwr import LGWR
from src.kernel.lgwr_kernel import LgwrKernel

def load_bandwidth_vector(bandwidth_str):
    bandwidth = ast.literal_eval(bandwidth_str)
    bandwidth = np.array([int(x) for x in bandwidth]).reshape(-1, 1)
    return bandwidth


def reload_logs(log_file_path: str):
    path = Path(log_file_path)
    with path.open("r", encoding="utf-8") as f:
        mode_info = json.load(f)
    bandwidth_vectors = mode_info['bandwidth_optimization']
    for v in bandwidth_vectors:
        v['bandwidth'] = load_bandwidth_vector(v['bandwidth'])
    return bandwidth_vectors


def reload_lgwr(dataset, bandwidth):
    kernel = LgwrKernel(
        dataset,
        kernel_type='bisquare',
        kernel_bandwidth_type='adaptive'
    )
    model = LGWR(dataset, kernel)

    
    model.update_local_bandwidth_vector(bandwidth["bandwidth"]).fit()
    return model
