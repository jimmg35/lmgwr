import numpy as np
import pandas as pd
from typing import List
from .interfaces.spatial_dataset import IDataPoint, IDataset, IFieldInfo
import logging
from pandas import DataFrame


class SpatialDataset(IDataset):
    """
    A class to represent a spatial dataset with associated field information.

    This class is responsible for verifying and managing spatial data points.
    It uses `fieldInfo` to interpret the structure of the dataset and extracts
    data points for spatial analysis.

    Attributes:
        dataPoints (List[IDataPoint] | None): A list of data points extracted from the dataset.
        fieldInfo (IFieldInfo | None): Information about the fields in the dataset, such as
                                       response and predictor fields, and coordinates.
        isSpherical (bool): A flag indicating whether the dataset is spherical or not.
    """
    dataPoints: List[IDataPoint] | None = None
    fieldInfo: IFieldInfo | None = None
    isSpherical: bool = False

    def __init__(self, data: DataFrame, fieldInfo: IFieldInfo, isSpherical: bool = False) -> None:
        """
        Initializes the SpatialDataset with the given data and field information.

        Verifies that the necessary fields are present in the dataset and creates data points.

        Args:
            data (DataFrame): The dataset containing all data points.
            fieldInfo (IFieldInfo): An object containing information about the dataset fields.

        Raises:
            ValueError: If the required fields are missing from the dataset.
        """
        self.fieldInfo = fieldInfo
        self._verify_fields(data)
        self.dataPoints = self._create_data_points(data)
        self.isSpherical = isSpherical

    def _verify_fields(self, data: pd.DataFrame) -> None:
        """
        Verifies that the dataset contains all required fields specified in `fieldInfo`.

        This method checks if the response field, coordinate fields, and predictor fields
        are present in the dataset. If any fields are missing, an error is raised.

        Args:
            data (pd.DataFrame): The dataset to be verified.

        Raises:
            ValueError: If `fieldInfo` is not set or if required fields are missing from the dataset.
        """
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
        """
        Creates a list of data points from the dataset.

        This method iterates over each row in the dataset, extracts the necessary fields,
        and creates `IDataPoint` objects for spatial analysis.

        Args:
            data (pd.DataFrame): The dataset containing all data points.

        Returns:
            List[IDataPoint]: A list of data points created from the dataset.

        Raises:
            ValueError: If `fieldInfo` is not set.
            Exception: If there is an error while creating data points.
        """
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


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/synthetic_dataset.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        IFieldInfo(
            predictor_fields=['temperature', 'moisture'],
            response_field='pm25',
            coordinate_x_field='coor_x',
            coordinate_y_field='coor_y'
        )
    )
