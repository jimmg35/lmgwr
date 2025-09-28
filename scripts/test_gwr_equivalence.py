"""Test for numerical parity between legacy mgwr.GWR and src.model.gwr.GWR."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset.interfaces.idataset import FieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR as NewGWR
from src.log.ilogger import ILogger

from mgwr import gwr as mgwr_gwr  # type: ignore[import]
from mgwr import sel_bw as mgwr_sel  # type: ignore[import]


RTOL = 1e-6
ATOL = 1e-9


class _NullLogger:
    """Minimal logger stub for tests."""

    def append_info(self, *args, **kwargs) -> None:  # noqa: D401 - simple stub
        pass

    def append_bandwidth_optimization(self, *args, **kwargs) -> None:
        pass

    def append_training_process(self, *args, **kwargs) -> None:
        pass

    def update_matrics(self, *args, **kwargs) -> None:
        pass


def _set_single_thread_environment() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    os.environ["PYTHONHASHSEED"] = "0"


def _assert_allclose(name: str, old: np.ndarray, new: np.ndarray) -> None:
    if old.shape != new.shape:
        pytest.fail(f"{name} shape mismatch: {old.shape} vs {new.shape}")

    if np.allclose(old, new, rtol=RTOL, atol=ATOL):
        print("{} within tolerance (rtol={}, atol={})".format(name, RTOL, ATOL))
        return

    diff = np.abs(old - new)
    tol = ATOL + RTOL * np.abs(old)
    mask = diff > tol

    if mask.ndim == 0:
        max_diff = float(diff)
        preview = [[0]]
    else:
        indices = np.argwhere(mask)
        preview = indices[:5].tolist()
        max_diff = float(diff.max())

    pytest.fail(
        f"{name} differ beyond tolerance (rtol={RTOL}, atol={ATOL}); "
        f"max diff {max_diff:.3e}; first differing indices {preview}"
    )


def test_gwr_equivalence() -> None:
    # mgwr_gwr = pytest.importorskip(
    #     "mgwr.gwr",
    #     reason="MGWR package is required for equivalence testing.",
    # )
    # mgwr_sel = pytest.importorskip(
    #     "mgwr.sel_bw",
    #     reason="MGWR bandwidth selector is required for equivalence testing.",
    # )
    OldGWR = mgwr_gwr.GWR
    Sel_BW = mgwr_sel.Sel_BW

    _set_single_thread_environment()
    np.random.seed(42)

    data_path = Path(__file__).resolve().parent.parent / "data" / "GData_utm.csv"
    data = pd.read_csv(data_path)

    field_info = FieldInfo(
        predictor_fields=["PctFB", "PctBlack", "PctRural"],
        response_field="PctBach",
        coordinate_x_field="X",
        coordinate_y_field="Y",
    )

    dataset = SpatialDataset(
        data,
        field_info,
        logger=None,
        isSpherical=False,
    )

    coords = np.asarray(dataset.coordinates, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.float64)
    X = np.asarray(dataset.X, dtype=np.float64)

    bandwidth_selector = Sel_BW(
        coords,
        y,
        X,
        fixed=False,
        kernel="bisquare",
        constant=False,
    )
    bandwidth_old = bandwidth_selector.search()
    if bandwidth_old is not None:
        bandwidth_old = int(round(bandwidth_old))

    legacy_model = OldGWR(
        coords,
        y,
        X,
        bandwidth_old,
        fixed=False,
        kernel="bisquare",
        constant=False,
    )
    legacy_results = legacy_model.fit()

    betas_old = np.asarray(legacy_results.params, dtype=np.float64)
    yhat_old = np.asarray(legacy_results.predy, dtype=np.float64)
    if yhat_old.ndim == 1:
        yhat_old = yhat_old.reshape(-1, 1)
    residuals_old = np.asarray(legacy_results.resid_response, dtype=np.float64)
    if residuals_old.ndim == 1:
        residuals_old = residuals_old.reshape(-1, 1)
    bandwidth_old_result = getattr(legacy_results, "bw", bandwidth_old)
    aicc_old = float(legacy_results.aicc)

    kernel = GwrKernel(dataset, "bisquare", "adaptive")
    kernel.weighted_matrix_cache = {}
    kernel.distance_vector_cache = {}
    kernel.update_bandwidth(bandwidth_old_result)

    # new_logger = _NullLogger()
    new_logger = ILogger()
    new_model = NewGWR(dataset, kernel, new_logger)
    new_model.fit()

    betas_new = np.asarray(new_model.betas, dtype=np.float64)
    yhat_new = np.asarray(new_model.y_hats, dtype=np.float64)
    if yhat_new.ndim == 1:
        yhat_new = yhat_new.reshape(-1, 1)
    residuals_new = np.asarray(new_model.residuals, dtype=np.float64)
    if residuals_new.ndim == 1:
        residuals_new = residuals_new.reshape(-1, 1)
    bandwidth_new = kernel.bandwidth
    aicc_new = float(new_model.aicc)

    _assert_allclose("beta", betas_old, betas_new)
    _assert_allclose("yhat", yhat_old, yhat_new)
    _assert_allclose("residuals", residuals_old, residuals_new)
    _assert_allclose(
        "bandwidth",
        np.asarray(bandwidth_old_result, dtype=np.float64),
        np.asarray(bandwidth_new, dtype=np.float64),
    )
    _assert_allclose(
        "AICc",
        np.asarray(aicc_old, dtype=np.float64),
        np.asarray(aicc_new, dtype=np.float64),
    )


test_gwr_equivalence()

