import numpy as np
import pandas as pd
import geopandas as gp
from mgwr.gwr import MGWR as OfficialMGWR
from mgwr.sel_bw import Sel_BW
from src.kernel.gwr_kernel import GwrKernel
from src.model.gwr import GWR
from src.model.mgwr import MGWR
from src.dataset.interfaces.idataset import FieldInfo
from src.dataset.spatial_dataset import SpatialDataset


# georgia_data = pd.read_csv(r'./data/GData_utm.csv')
# # georgia_geometry = gp.read_file(r'../../../data/G_utm.shp')
# dataset = SpatialDataset(
#     georgia_data,
#     FieldInfo(
#         predictor_fields=['PctFB', 'PctBlack', 'PctRural'],
#         response_field='PctBach',
#         coordinate_x_field='X',
#         coordinate_y_field='Y'
#     ),
#     isSpherical=False,
#     # geometry=georgia_geometry
# )
# # dataset.plot_map()


georgia_data = pd.read_csv(r'./base_dataset_exclude_south.csv')
dataset = SpatialDataset(
    georgia_data,
    FieldInfo(
        predictor_fields=[
            "pct_65_Z_SCORE",
            "pct_18_29_Z_SCORE",
            "pct_male_Z_SCORE",
            "pct_black_Z_SCORE",
            "pct_american_indian_Z_SCORE",
            "pct_hispanic_Z_SCORE",
            "pct_asian_Z_SCORE",
            "pct_bach_Z_SCORE",
            "pct_smoking_Z_SCORE",
            "median_income_Z_SCORE",
            "population_density_Z_SCORE"
        ],
        response_field='pct_yes_Z_SCORE',
        coordinate_x_field='longitude',
        coordinate_y_field='latitude'
    ),
    isSpherical=True
)


g_X = dataset.X[:, 1:]
g_y = dataset.y.reshape(-1, 1)
g_coords = dataset.coordinates.tolist()

mgwr_selector = Sel_BW(
    g_coords, 
    g_y, 
    g_X, 
    multi=True, 
    # spherical=True
)
official_mgwr_bw = mgwr_selector.search(multi_bw_min=[2])
official_mgwr_results = OfficialMGWR(
    g_coords, 
    g_y, 
    g_X, 
    mgwr_selector, 
    hat_matrix=True, 
    # spherical=True
).exact_fit()

official_mgwr_results.summary()
print("=========== Best Bandwidths (MGWR) ==============")
print(official_mgwr_bw)


print("=========== running Monte Carlo ==============")
ps = official_mgwr_results.spatial_variability(mgwr_selector)

print("=========== p-values of monte carlo (MGWR) ==============")
print(ps)