from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List
from pandas import DataFrame


@dataclass
class IDataPoint:
    y: float
    X: np.ndarray
    coordinate_x: float
    coordinate_y: float


@dataclass
class IFieldInfo:
    predictor_fields: List[str]
    response_field: str
    coordinate_x_field: str
    coordinate_y_field: str


@dataclass
class IDataset:
    dataPoints: List[IDataPoint] | None
    fieldInfo: IFieldInfo | None

    def _verify_fields(self, data: DataFrame) -> None:
        raise NotImplementedError()

    def _create_data_points(self, data: pd.DataFrame) -> List[IDataPoint]:
        raise NotImplementedError()
