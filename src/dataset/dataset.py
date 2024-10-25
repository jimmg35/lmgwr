import numpy as np
import pandas as pd
from typing import List
from .interfaces.interface import IDataPoint, ISpatialDatasetParams, IDataset, IFieldInfo
import logging


class Dataset(IDataset):
    dataPoints: List[IDataPoint] | None = None
    fieldInfo: IFieldInfo | None = None

    def __init__(self, param: ISpatialDatasetParams):
        self.fieldInfo = param.fieldInfo
        self._verify_fields(param.data)
        self.dataPoints = self._create_data_points(param.data)

    def _verify_fields(self, data: pd.DataFrame) -> None:
        if self.fieldInfo is None:
            raise ValueError("FieldInfo is not set")
        required_fields = [
            self.fieldInfo.response_field,
            self.fieldInfo.coordinate_x_field,
            self.fieldInfo.coordinate_y_field,
            *self.fieldInfo.predictor_fields
        ]
        missing_fields = [
            field for field in required_fields if field not in data.columns]
        if missing_fields:
            raise ValueError(
                f"Missing fields in the dataset: {', '.join(missing_fields)}"
            )
        logging.info(f"$ Required fields are present in the dataset.")

    def _create_data_points(self, data: pd.DataFrame) -> List[IDataPoint]:
        if self.fieldInfo is None:
            raise ValueError("FieldInfo is not set")
        try:
            data_points = []
            for _, row in data.iterrows():
                y = row[self.fieldInfo.response_field]
                X = row[self.fieldInfo.predictor_fields].values
                coordinate_x = row[self.fieldInfo.coordinate_x_field]
                coordinate_y = row[self.fieldInfo.coordinate_y_field]
                data_points.append(IDataPoint(y=y, X=np.array(
                    X), coordinate_x=coordinate_x, coordinate_y=coordinate_y))
            logging.info(f"$ Data points created successfully.")
            return data_points
        except Exception as e:
            logging.error(f"Error creating data points: {e}")
            raise e
