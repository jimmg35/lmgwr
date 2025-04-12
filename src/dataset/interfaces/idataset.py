from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List
from pandas import DataFrame


@dataclass
class FieldInfo:
    predictor_fields: List[str]
    response_field: str
    coordinate_x_field: str
    coordinate_y_field: str


@dataclass
class IDataset:
    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    coordinates: npt.NDArray[np.float64]

    def __len____(self) -> int:
        """
        Returns the number of data points in the dataset.
        """
        raise NotImplementedError("Subclasses should implement this!")
