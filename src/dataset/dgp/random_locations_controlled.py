import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ============================================================
# 1. Grid and covariance
# ============================================================

def make_grid(n=60, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
    """
    Create an n x n regular grid in [xlim] x [ylim].

    Returns:
        X, Y: 2D arrays of shape (n, n)
        coords: (n*n, 2) array of (x, y) pairs, row-major order
    """
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    return X, Y, coords

def squared_exponential_cov(coords, length_scale, sigma=1.0, nugget=1e-6):
    """
    Squared-exponential (RBF) covariance matrix for a set of 2D coordinates.

    k(r) = sigma^2 * exp(-0.5 * (r / length_scale)^2)

    Args:
        coords: (N, 2) array of point coordinates
        length_scale: correlation length (controls smoothness)
        sigma: marginal standard deviation
        nugget: small diagonal term for numerical stability

    Returns:
        K: (N, N) covariance matrix
    """
    dists = cdist(coords, coords)  # pairwise Euclidean distances
    K = sigma**2 * np.exp(-0.5 * (dists / length_scale)**2)
    K += nugget * np.eye(len(coords))
    return K

# ============================================================
# 2. One shared noise vector ε, many GRFs
# ============================================================

def draw_base_noise(n_points, seed=123):
    """
    Draw a single N(0,1)^N noise vector ε using a fixed seed.

    This ε will be reused for all GRFs so they share the
    same underlying random pattern structure.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_points)
    return eps

def grf_from_eps(coords, eps, length_scale, sigma=1.0):
    """
    Create a GRF realization from a fixed ε and a given length-scale.

    Mathematical idea:
      Given ε ~ N(0, I_N),
      K_ℓ = covariance matrix with length-scale ℓ,
      L_ℓ = Cholesky factor of K_ℓ such that K_ℓ = L_ℓ L_ℓ^T,
      then Z_ℓ = L_ℓ ε ~ N(0, K_ℓ).

    Using the SAME ε but different ℓ:
      - Z_smooth and Z_rough share the same underlying random "pattern"
        but appear smoother/rougher depending on ℓ.
    """
    K = squared_exponential_cov(coords, length_scale, sigma)
    L = np.linalg.cholesky(K)
    z = L @ eps  # shape (N,)
    return z

# ============================================================
# 3. Random “blob” weight field w(x, y)
# ============================================================

def random_blob_weight_field(X, Y,
                             n_patches=6,
                             r_min_cells=8,
                             r_max_cells=20,
                             power=2.0,
                             random_state=0):
    """
    Construct a smooth weight field w(x, y) in [0,1] using multiple radial blobs.

    Steps:
      1. Choose random centers (cx, cy) in the domain.
      2. For each center, choose a random radius r in grid units.
      3. For each grid cell, compute distance d to each center.
      4. Define a bump: b = max(1 - (d/r)^2, 0)^power.
      5. Take the maximum over all blobs to obtain w(x,y).

    Interpretation:
      - w ≈ 1: "short-scale" GRF will dominate.
      - w ≈ 0: "long-scale" GRF will dominate.
      - Smooth transitions create natural, irregular regions.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    w = np.zeros_like(X)

    for _ in range(n_patches):
        # random center
        cx, cy = rng.uniform(0.0, 1.0, size=2)

        # random radius in cells, then convert to domain units
        r_cells = rng.integers(r_min_cells, r_max_cells + 1)
        r = r_cells / (n - 1)

        d = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # polynomial bump: 1 at center, 0 at distance >= r
        bump = 1.0 - (d / r)**2
        bump[bump < 0] = 0.0
        bump = bump**power

        # combine blobs via max (union of patches)
        w = np.maximum(w, bump)

    return np.clip(w, 0.0, 1.0)

# ============================================================
# 4. Processes
# ============================================================

def process1_homogeneous_scale(n=60,
                               length_scale=0.3,
                               sigma=1.0,
                               seed_eps=123):
    """
    Process 1: Homogeneous-scale GRF
      - One global length-scale ℓ_global.
      - Rate of change is uniform across space.

    Uses the shared ε, so that you can compare this
    to other processes built from the same ε.
    """
    X, Y, coords = make_grid(n=n)
    eps = draw_base_noise(coords.shape[0], seed=seed_eps)
    z = grf_from_eps(coords, eps, length_scale=length_scale, sigma=sigma)
    Z = z.reshape(n, n)
    return X, Y, Z, coords, eps

def process2_composite_random_blobs(n=60,
                                    length_scale_short=0.08,
                                    length_scale_long=0.4,
                                    sigma=1.0,
                                    n_patches=6,
                                    r_min_cells=8,
                                    r_max_cells=20,
                                    seed_eps=123,
                                    seed_blobs=0):
    """
    Process 2: Composite-scale process with random blob regions.

    1. Use the SAME ε as Process 1 (same seed_eps).
    2. Build:
         Z_short = GRF with small length-scale (high-frequency)
         Z_long  = GRF with large length-scale (smooth)
       Both are L_ℓ * ε with different ℓ, so they share the same
       underlying pattern but appear with different rates of change.

    3. Compute a random weight field w(x,y) in [0,1] via blobs.
    4. Composite field:
         Z_comp = w * Z_short + (1 - w) * Z_long

       → Regions with large w behave like short-scale,
         regions with small w behave like long-scale.
    """
    X, Y, coords = make_grid(n=n)

    # shared ε
    eps = draw_base_noise(coords.shape[0], seed=seed_eps)

    # two GRFs with SAME ε but different length-scales
    z_short = grf_from_eps(coords, eps, length_scale=length_scale_short, sigma=sigma)
    z_long  = grf_from_eps(coords, eps, length_scale=length_scale_long,  sigma=sigma)

    Z_short = z_short.reshape(n, n)
    Z_long  = z_long.reshape(n, n)

    # irregular weight field
    W = random_blob_weight_field(
        X, Y,
        n_patches=n_patches,
        r_min_cells=r_min_cells,
        r_max_cells=r_max_cells,
        power=2.0,
        random_state=seed_blobs
    )

    # composite field
    Z_comp = W * Z_short + (1.0 - W) * Z_long

    return X, Y, Z_comp, Z_short, Z_long, W, coords, eps

# ============================================================
# 5. Demo
# ============================================================

if __name__ == "__main__":
    # Process 1: homogeneous
    X1, Y1, Z_h, coords1, eps1 = process1_homogeneous_scale(
        n=60,
        length_scale=0.3,
        sigma=1.0,
        seed_eps=123
    )

    # Process 2: composite
    X2, Y2, Z_comp, Z_short, Z_long, W, coords2, eps2 = process2_composite_random_blobs(
        n=60,
        length_scale_short=0.08,
        length_scale_long=0.4,
        sigma=1.0,
        n_patches=7,
        r_min_cells=6,
        r_max_cells=18,
        seed_eps=123,      # <-- SAME ε as Process 1
        seed_blobs=0
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axes[0, 0].imshow(Z_h, origin="lower", extent=[0, 1, 0, 1])
    axes[0, 0].set_title("Process 1: homogeneous-scale GRF")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(W, origin="lower", extent=[0, 1, 0, 1])
    axes[0, 1].set_title("Weight field w(x, y) (short-scale dominance)")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(Z_short, origin="lower", extent=[0, 1, 0, 1])
    axes[1, 0].set_title("Short-scale GRF (same ε, smaller ℓ)")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(Z_comp, origin="lower", extent=[0, 1, 0, 1])
    axes[1, 1].set_title("Process 2: composite-scale field")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()
