import pandas as pd


from src.dataset.spatial_dataset import IFieldInfo
from src.dataset.spatial_dataset import SpatialDataset
from src.dataset.interfaces.spatial_dataset import IFieldInfo
from src.model.gwr import GWR
from src.kernel.gwr_kernel import GwrKernel
from src.optimizer.gwr_optimizer import GwrOptimizer
from src.log.gwr_logger import GwrLogger


def fit_OLS(dataset):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    model = LinearRegression()
    model.fit(dataset.X, dataset.y)
    y_pred = model.predict(dataset.X)
    return r2_score(dataset.y, y_pred)


def fit_GWR(dataset, logger):
    kernel = GwrKernel(dataset, logger, 'bisquare')
    gwr = GWR(dataset, kernel, logger)
    optimizer = GwrOptimizer(gwr, kernel, logger)
    optimal_bandwidth = optimizer.optimize()
    return gwr.r_squared, optimal_bandwidth


if __name__ == '__main__':

    georgia_dataframe = pd.read_csv(r'./data/GData_utm.csv')

    gwrLogger = GwrLogger()
    dataset = SpatialDataset(
        georgia_dataframe,
        IFieldInfo(
            predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
            response_field='PctBach',
            coordinate_x_field='X',
            coordinate_y_field='Y'
        ),
        gwrLogger,
        isSpherical=False
    )

    ols_r2 = fit_OLS(dataset)
    gwr_r2, optimal_bandwidth = fit_GWR(dataset, gwrLogger)

    print("====================================")
    print(f"OLS R2: {ols_r2}")
    print(f"GWR R2: {gwr_r2} | Optimal bandwidth: {optimal_bandwidth}")
