
# from spglm.glm import GLM, GLMResults
from ..dataset.interfaces.spatial_dataset import IDataset
from ..dataset.spatial_dataset import SpatialDataset


class GWR():
    """
    A class for performing Geographically Weighted Regression (GWR) using a spatial dataset.

    Attributes:
        dataset (SpatialDataset): The spatial dataset used for the GWR analysis.
    """

    dataset: SpatialDataset | None = None

    def __init__(self, dataset: SpatialDataset) -> None:
        """
        Initializes the GWR model with the given dataset.

        Args:
            dataset (SpatialDataset): The spatial dataset to be used for the GWR model.
        """
        self.dataset = dataset

    def fit(self) -> None:
        """
        Fit the GWR model with the provided dataset and specified bandwidth.

        This method should implement the algorithm to fit the GWR model to the dataset.

        Raises:
            NotImplementedError: If the method is not yet implemented.
        """
        raise NotImplementedError("Method not implemented yet")

    def __build_weight_matrix(self, index: int, bandwidth: float) -> None:
        """
        Build the spatial weight matrix for a particular datapoint.

        This method should create a weight matrix that reflects the spatial relationship 
        between data points, typically based on geographic proximity.

        Args:
            index (int): The index of the target data point.
            bandwidth (float): The bandwidth value for calculating weights (it can be manually configured).

        Raises:
            NotImplementedError: If the method is not yet implemented.
        """
        raise NotImplementedError("Method not implemented yet")
