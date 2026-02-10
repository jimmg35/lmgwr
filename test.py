import numpy as np
import plotly.graph_objects as go

# =====================================================
# 1. Generate irregular surface z = f(x, y)
# =====================================================
nx, ny = 160, 160
x = np.linspace(-3.2, 3.2, nx)
y = np.linspace(-3.2, 3.2, ny)
X, Y = np.meshgrid(x, y, indexing="xy")

def surface(X, Y):
    bump1 = 1.8 * np.exp(-((X + 1.2)**2 + (Y - 0.4)**2) / 0.55)
    bump2 = -1.3 * np.exp(-((X - 0.9)**2 + (Y + 1.0)**2) / 0.35)
    ridge = 0.9 * np.exp(-(Y**2) / 0.9) * np.tanh(1.2 * (X - 0.2))
    waves = 0.35 * np.sin(1.8 * X) * np.cos(1.2 * Y)
    tilt  = 0.10 * X - 0.06 * Y
    return bump1 + bump2 + ridge + waves + tilt

Z = surface(X, Y)

dx = x[1] - x[0]
dy = y[1] - y[0]

# =====================================================
# 2. Finite difference Hessian (central differences)
# =====================================================
Zxx = (Z[:, 2:] - 2 * Z[:, 1:-1] + Z[:, :-2]) / dx**2
Zyy = (Z[2:, :] - 2 * Z[1:-1, :] + Z[:-2, :]) / dy**2
Zxy = (Z[2:, 2:] - Z[2:, :-2] - Z[:-2, 2:] + Z[:-2, :-2]) / (4 * dx * dy)

# interior alignment
Xi = X[1:-1, 1:-1]
Yi = Y[1:-1, 1:-1]
Zi = Z[1:-1, 1:-1]

Zxx_i = Zxx[1:-1, :]
Zyy_i = Zyy[:, 1:-1]
Zxy_i = Zxy

# Frobenius norm of Hessian
hess_norm = np.sqrt(Zxx_i**2 + 2 * Zxy_i**2 + Zyy_i**2)

# =====================================================
# 3. Select showcase points (curvature-based ONLY)
# =====================================================
# A: Very low curvature (near-linear)
low_idx = np.unravel_index(
    np.argmin(hess_norm),
    hess_norm.shape
)

# B: Moderate curvature (median)
mid_target = np.quantile(hess_norm, 0.5)
mid_idx = np.unravel_index(
    np.argmin(np.abs(hess_norm - mid_target)),
    hess_norm.shape
)

# C: High curvature (top 1%, robust)
high_target = np.quantile(hess_norm, 0.99)
high_idx = np.unravel_index(
    np.argmin(np.abs(hess_norm - high_target)),
    hess_norm.shape
)

points = {
    "Low curvature (slow rate of change)": low_idx,
    "Moderate curvature": mid_idx,
    "High curvature (rapid rate of change)": high_idx
}

# =====================================================
# 3.5 Print Hessian diagnostics (SHOWCASE)
# =====================================================
print("\nCurvature-based showcase points:\n")

for name, (i, j) in points.items():
    H_xx = Zxx_i[i, j]
    H_xy = Zxy_i[i, j]
    H_yy = Zyy_i[i, j]
    frob = hess_norm[i, j]

    print(name)
    print(f"  Location (x, y) = ({Xi[i, j]:.3f}, {Yi[i, j]:.3f})")
    print("  Hessian:")
    print(f"    [ {H_xx:+.4f}  {H_xy:+.4f} ]")
    print(f"    [ {H_xy:+.4f}  {H_yy:+.4f} ]")
    print(f"  Frobenius norm = {frob:.4f}\n")

# =====================================================
# 4. Plotly 3D visualization
# =====================================================
fig = go.Figure()

fig.add_trace(go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale="Viridis",
    opacity=0.9,
    showscale=True
))

for name, (i, j) in points.items():
    fig.add_trace(go.Scatter3d(
        x=[Xi[i, j]],
        y=[Yi[i, j]],
        z=[Zi[i, j]],
        mode="markers+text",
        marker=dict(size=7, color="red"),
        text=[name],
        textposition="top center",
        name=name
    ))

fig.update_layout(
    title="Curvature-based showcase: rate of change intensity",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# =====================================================
# 5. Save to HTML
# =====================================================
fig.write_html("curvature_showcase.html")
print("Saved: curvature_showcase.html")





# ==================================================
# LGWR code

from src.dataset.simulated_spatial_dataset import SimulatedSpatialDataset
from src.model.lgwr import LGWR
from src.kernel.lgwr_kernel import LgwrKernel

from mgwr.sel_bw import Sel_BW

k = 1
field_size = 40
dataset = SimulatedSpatialDataset(
    k=k,
    field_size=field_size,
    useIntercept=False
)
[X] = dataset.generate_data()
[beta] = dataset.generate_processes()
[y, err] = dataset.fit_y(X, beta)


## Find the global optimal bandwidth
g_X = dataset.X[:, 1:]
g_y = dataset.y.reshape(-1, 1)
g_coords = dataset.coordinates.tolist()
gwr_selector = Sel_BW(
    g_coords,
    g_y,
    g_X
)
global_optimal_bandwidth = gwr_selector.search(bw_min=2)

###############################
#                             #
#  Initialize the LGWR model  #
#                             #
###############################
kernel = LgwrKernel(
    dataset,
    kernel_type='bisquare',
    kernel_bandwidth_type='adaptive'
)
lgwr = LGWR(dataset, kernel)

pilot_bandwidth = global_optimal_bandwidth

lgwr.update_local_bandwidth_vector(np.full(dataset.n, pilot_bandwidth))
lgwr.fit()