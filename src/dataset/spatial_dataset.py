
# import torch
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt

from src.dataset.interfaces.spatial_dataset import IDataset, FieldInfo
from src.log.ilogger import ILogger


class SpatialDataset(IDataset):
    """
    A class to represent a spatial dataset with associated field information.

    This class manages spatial data points and validates the structure and
    completeness of the dataset based on the provided `fieldInfo`. It extracts
    data points for spatial analysis and structures them in a format suitable
    for geospatial modeling.

    Attributes:
        fieldInfo (FieldInfo | None): Information about the fields in the dataset, such as
                                       response and predictor fields, and coordinate fields.
        isSpherical (bool): A flag indicating whether the dataset is in spherical coordinates.
        X (npt.NDArray[np.float64]): A matrix of predictor values extracted from the data points.
        y (npt.NDArray[np.float64]): A column vector of response values extracted from the data points.
    """
    logger: ILogger | None = None
    fieldInfo: FieldInfo
    isSpherical: bool = False
    useIntercept: bool = True
    isStandardize: bool = True

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    coordinates: npt.NDArray[np.float64]
    geometry: GeoDataFrame | None = None

    def __init__(
        self,
        data: DataFrame,
        fieldInfo: FieldInfo,
        logger: ILogger | None = None,
        geometry: GeoDataFrame | None = None,
        isSpherical: bool = False,
        useIntercept: bool = True,
        isStandardize: bool = True,
    ) -> None:
        """
        Initializes the SpatialDataset with provided data and field information.

        This method verifies the presence of necessary fields in the dataset and then
        initializes the data points for analysis, extracting predictors and response
        values into structured matrices.

        Args:
            data (DataFrame): The dataset containing spatial data points.
            fieldInfo (FieldInfo): An object containing metadata about the dataset fields,
                                    including response, predictor, and coordinate fields.
            isSpherical (bool): Whether the dataset uses spherical coordinates. Defaults to False.

        Raises:
            ValueError: If any required fields specified in `fieldInfo` are missing from the dataset.
        """
        # Register the state of the dataset
        if logger is not None:
            self.logger = logger
        self.geometry = geometry
        self.fieldInfo = fieldInfo
        self.isSpherical = isSpherical
        self.useIntercept = useIntercept
        self.isStandardize = isStandardize

        # verify the fields in the dataset
        self.__verify_data_schema(data)
        self.__prepare_data(data)

    def __len__(self) -> int:
        """
        Returns the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self.X)

    def __prepare_data(self, data: pd.DataFrame):
        """
        Prepares the dataset by extracting predictor and response variables, and coordinates.

        Args:
            data (pd.DataFrame): The dataset containing spatial data points.
        """
        # Transforming the data into matrix form.
        self.X = data[self.fieldInfo.predictor_fields].to_numpy(
            dtype=np.float64)
        self.y = data[self.fieldInfo.response_field].to_numpy(dtype=np.float64)
        self.coordinates = data[[self.fieldInfo.coordinate_x_field,
                                 self.fieldInfo.coordinate_y_field]].to_numpy(dtype=np.float64)

        # Standardize the columns
        self.X = (self.X - self.X.mean(axis=0)
                  ) / self.X.std(axis=0)
        self.y = self.y.reshape((-1, 1))
        self.y = (self.y - self.y.mean(axis=0)) / self.y.std(axis=0)

        # Add a column of ones as the first column for the intercept
        if self.useIntercept:
            self.X = np.hstack(
                (np.ones((self.X.shape[0], 1)), self.X)
            )

    def __verify_data_schema(self, data: pd.DataFrame) -> None:
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

        if self.logger is not None:
            self.logger.append_info(
                f"{self.__class__.__name__} : Data schema is verified.")

    def plot_map(self):
        if self.geometry is None:
            raise ValueError(
                "Geometry is not set, please provide a GeoDataFrame"
            )

        fig, ax = plt.subplots(figsize=(10, 10))
        self.geometry.plot(ax=ax, edgecolor='black', facecolor='white')
        self.geometry.centroid.plot(ax=ax, c='black')


if __name__ == '__main__':
    synthetic_data = pd.read_csv(r'./data/synthetic_dataset.csv')

    spatialDataset = SpatialDataset(
        synthetic_data,
        FieldInfo(
            predictor_fields=['temperature', 'moisture'],
            response_field='pm25',
            coordinate_x_field='coor_x',
            coordinate_y_field='coor_y'
        )
    )
