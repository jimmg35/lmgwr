import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List
from pandas import DataFrame
from src.dataset.interfaces.spatial_dataset import IDataPoint, IDataset, IFieldInfo


class SpatialDataset(IDataset):
    """
    A class to represent a spatial dataset with associated field information.

    This class manages spatial data points and validates the structure and
    completeness of the dataset based on the provided `fieldInfo`. It extracts
    data points for spatial analysis and structures them in a format suitable
    for geospatial modeling.

    Attributes:
        dataPoints (List[IDataPoint] | None): A list of data points extracted from the dataset.
        fieldInfo (IFieldInfo | None): Information about the fields in the dataset, such as
                                       response and predictor fields, and coordinate fields.
        isSpherical (bool): A flag indicating whether the dataset is in spherical coordinates.
        x_matrix (npt.NDArray[np.float64]): A matrix of predictor values extracted from the data points.
        y (npt.NDArray[np.float64]): A column vector of response values extracted from the data points.
    """

    dataPoints: List[IDataPoint] | None = None
    fieldInfo: IFieldInfo | None = None
    isSpherical: bool = False
    x_matrix: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, data: DataFrame, fieldInfo: IFieldInfo, isSpherical: bool = False) -> None:
        """
        Initializes the SpatialDataset with provided data and field information.

        This method verifies the presence of necessary fields in the dataset and then
        initializes the data points for analysis, extracting predictors and response
        values into structured matrices.

        Args:
            data (DataFrame): The dataset containing spatial data points.
            fieldInfo (IFieldInfo): An object containing metadata about the dataset fields,
                                    including response, predictor, and coordinate fields.
            isSpherical (bool): Whether the dataset uses spherical coordinates. Defaults to False.

        Raises:
            ValueError: If any required fields specified in `fieldInfo` are missing from the dataset.
        """
        self.fieldInfo = fieldInfo
        self._verify_fields(data)
        self.dataPoints = self._create_data_points(data)
        self.isSpherical = isSpherical
        self.x_matrix = np.vstack(
            [data_point.X for data_point in self.dataPoints])
        self.y = np.array([[data_point.y] for data_point in self.dataPoints])

    def _verify_fields(self, data: pd.DataFrame) -> None:
        """
        Verifies the presence of all required fields in the dataset based on `fieldInfo`.

        Checks if the response field, coordinate fields, and predictor fields specified in
        `fieldInfo` exist within the dataset columns. If any required fields are missing,
        raises a ValueError with details about the missing fields.

        Args:
            data (pd.DataFrame): The dataset to validate.

        Raises:
            ValueError: If `fieldInfo` is not set or if required fields are missing from the dataset.
        """
        if self.fieldInfo is None:
            raise ValueError("FieldInfo is not set")

        # Gather required fields from fieldInfo
        required_fields = [
            self.fieldInfo.response_field,
            self.fieldInfo.coordinate_x_field,
            self.fieldInfo.coordinate_y_field,
            *self.fieldInfo.predictor_fields
        ]

        # Identify any fields missing from the dataset
        missing_fields = [
            field for field in required_fields if field not in data.columns]
        if missing_fields:
            raise ValueError(
                f"Missing fields in the dataset: {', '.join(missing_fields)}"
            )

        logging.info("All required fields are present in the dataset.")

    def _create_data_points(self, data: pd.DataFrame) -> List[IDataPoint]:
        """
        Creates and populates a list of data points from the dataset for spatial analysis.

        Iterates over each row in the dataset, extracts the response and predictor values
        along with the coordinate fields, and creates `IDataPoint` objects for each row.

        Args:
            data (pd.DataFrame): The dataset containing spatial data points.

        Returns:
            List[IDataPoint]: A list of `IDataPoint` objects created from each row of the dataset.

        Raises:
            ValueError: If `fieldInfo` is not set.
            Exception: If an error occurs while creating data points.
        """
        if self.fieldInfo is None:
            raise ValueError("FieldInfo is not set")

        try:
            data_points = []
            for _, row in data.iterrows():
                # Extract response, predictors, and coordinates from the row
                y = row[self.fieldInfo.response_field]
                X = row[self.fieldInfo.predictor_fields].values
                coordinate_x = row[self.fieldInfo.coordinate_x_field]
                coordinate_y = row[self.fieldInfo.coordinate_y_field]

                # Create an IDataPoint with extracted values
                data_points.append(IDataPoint(y=y, X=np.array(
                    X), coordinate_x=coordinate_x, coordinate_y=coordinate_y))

            logging.info("Data points created successfully.")
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
