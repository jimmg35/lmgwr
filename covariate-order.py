import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mgwr.sel_bw import Sel_BW
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import itertools

# Set this to True in the main block to run the summary and plot
SUMMARY_DATA = False 

def generate_data(N=1000, seed=42):
    """
    Generates simulated data with 6 covariate surfaces (X1 to X6) and 
    corresponding true parameter surfaces (b1 to b6) with distinct 
    levels of spatial heterogeneity, based on extensions of Li et al. (2020).
    
    Parameters:
    N (int): The number of observations to generate.
    seed (int): A random seed for reproducible results.
    
    Returns:
    tuple: A tuple containing:
        - coords (list): List of (u, v) coordinate tuples for MGWR.
        - y (np.ndarray): The response variable, shape (N, 1).
        - X (np.ndarray): The covariate matrix, shape (N, 6).
        - true_betas_df (pd.DataFrame): DataFrame with coordinates and
                                          true beta values for plotting.
    """
    np.random.seed(seed)
    
    # --- 1. Generate 1000 randomly distributed coordinates (u, v) in a circle ---
    r_rand = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    
    u = 12.5 + 12.5 * np.sqrt(r_rand) * np.cos(theta)
    v = 12.5 + 12.5 * np.sqrt(r_rand) * np.sin(theta)
    
    # Coordinates for calculation
    coords_u_v = np.stack([u, v], axis=1)
    
    # --- 2. Generate true parameter surfaces (True Betas) ---
    
    # b1: High heterogeneity (Central peak) - Li et al. Eq (8)
    b1 = 1 + 1/324 * (36 - (6 - u/2)**2) * (36 - (6 - v/2)**2)
    
    # b2: Low heterogeneity (Linear SW-NE trend) - Li et al. Eq (9)
    b2 = 2 + 1/24 * (u + v)
    
    # b3: Medium Heterogeneity (W-E Sinusoidal/Wave pattern)
    b3 = 3 + 0.5 * np.sin(u / 3) + 0.3 * np.cos(v / 5)
    
    # b4: Very Low Heterogeneity (Gentle global trend)
    # This surface uses the distance from the center (12.5, 12.5) for a very smooth gradient
    center_dist = np.sqrt((u - 12.5)**2 + (v - 12.5)**2)
    b4 = 4 + 0.05 * center_dist 

    # b5: Low-Medium Heterogeneity (Circular ring pattern)
    # Ring pattern (difference from a center distance, squared for symmetry)
    b5 = 5 + 0.1 * (center_dist - 6)**2
    
    # b6: Global (Constant coefficient) - Should converge to the largest bandwidth
    b6 = np.full(N, 6.0)

    # --- 3. Generate covariates (X1 to X6) and noise ---
    
    # All X variables drawn from a standard normal distribution N(0, 1)
    x1 = np.random.normal(0, 1, N)
    x2 = np.random.normal(0, 1, N)
    x3 = np.random.normal(0, 1, N)
    x4 = np.random.normal(0, 1, N)
    x5 = np.random.normal(0, 1, N)
    x6 = np.random.normal(0, 1, N)
    
    # Epsilon (noise) drawn from standard normal N(0, 1)
    epsilon = np.random.normal(0, 1, N)
    
    # --- 4. Generate response variable Y ---
    # y = b1*x1 + b2*x2 + b3*x3 + b4*x4 + b5*x5 + b6*x6 + epsilon
    y = b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 + b5 * x5 + b6 * x6 + epsilon
    
    # --- 5. Format data for MGWR and plotting ---
    coords = list(zip(u, v))
    X = np.stack([x1, x2, x3, x4, x5, x6], axis=1) # Shape (N, 6)
    y = y.reshape(-1, 1)           # Shape (N, 1)
    
    # Also return true betas for plotting/analysis
    true_betas_df = pd.DataFrame({
        'u': u, 'v': v, 
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6
    })
    
    return coords, y, X, true_betas_df

def plot_betas(true_betas_df):
    """
    Plots the six true beta parameter surfaces.
    
    Parameters:
    true_betas_df (pd.DataFrame): DataFrame containing 'u', 'v', 'b1' to 'b6'.
    """
    
    # Create a 2x3 subplot layout for 6 variables
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten() # Flatten the 2x3 array of axes for easy iteration
    
    beta_names = [f'b{i}' for i in range(1, 7)]
    titles = [
        'b1: High Heterogeneity (Peak)',
        'b2: Low Heterogeneity (Linear Trend)',
        'b3: Medium Heterogeneity (Wave)',
        'b4: Very Low Heterogeneity (Smooth)',
        'b5: Low-Medium Heterogeneity (Ring)',
        'b6: Global (Constant)'
    ]
    
    for i, ax in enumerate(axes):
        # Use 'viridis' for high contrast and color blindness friendly
        sc = ax.scatter(true_betas_df['u'], true_betas_df['v'], 
                        c=true_betas_df[beta_names[i]], cmap='viridis', s=20)
        
        # Add color bar next to the plot
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical')
        cbar.set_label(f'Beta {i+1} Coefficient Value')
        
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('u coordinate')
        ax.set_ylabel('v coordinate')
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('True Parameter Surfaces for 6 Covariates', fontsize=22, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save the figure
    plt.savefig("simulated_6betas_plot.png", dpi=150)
    print("\nPlot saved to 'simulated_6betas_plot.png'")
    
    # Show the plot
    plt.show()

def run_permutation_test(coords, y, X):
    """
    Iterates through all 6! permutations of the covariate matrix X 
    and finds the optimal bandwidth set for each order, EXCLUDING the Intercept.
    
    Parameters:
    coords (list): List of coordinates.
    y (np.ndarray): Standardized response variable.
    X (np.ndarray): Standardized covariate matrix, shape (N, 6).
    
    Returns:
    pd.DataFrame: A DataFrame summarizing the optimal bandwidths found 
                  for each variable order.
    """
    
    # The covariates (X) have 6 columns (0 to 5)
    n_covariates = X.shape[1]
    # Create the index of covariates [0, 1, 2, 3, 4, 5]
    covariate_indices = list(range(n_covariates))
    
    # Generate all possible permutations (6! = 720)
    all_permutations = list(itertools.permutations(covariate_indices))
    
    results_list = []
    
    print(f"Total number of permutations to test: {len(all_permutations)}")
    
    # Define the base variable names for tracking (matching b1 to b6)
    base_names = [f'X{idx+1}' for idx in range(n_covariates)]
    
    for i, permutation in enumerate(all_permutations):
        
        print(f"\nTesting permutation {i + 1}/{len(all_permutations)}: {permutation}")

        # 1. Reorder the X matrix according to the permutation
        # This changes the Backfitting sequence: X_ordered[0], X_ordered[1], ...
        X_ordered = X[:, permutation]
        
        # 2. Determine the variable names for easier tracking
        # permutation = (3, 0, 5, 2, 1, 4) -> ordered_names = ['X4', 'X1', 'X6', 'X3', 'X2', 'X5']
        ordered_names = [base_names[idx] for idx in permutation]
        
        # 3. Calibrate the MGWR model for this specific order
        try:
            # *** CRITICAL: Set constant=False to exclude the intercept ***
            mgwr_selector = Sel_BW(
                coords, 
                y, 
                X_ordered, 
                multi=True, 
                constant=False 
            )
            
            # Search for the optimal bandwidths
            # multi_bw_min=[2] is a safe minimum bandwidth
            mgwr_bw = mgwr_selector.search(
                multi_bw_min=[2] * n_covariates, # Provide a list of mins for all covariates
                verbose=False 
            )
            
            # Since constant=False, the number of returned BWs should match n_covariates (6)
            bws = list(mgwr_bw)
            
            # Create a dictionary to store the results
            result = {'Order_ID': i + 1}
            # Record the sequence of variables tested in this iteration
            result['Order_Sequence'] = str(ordered_names)
            
            # Store the bandwidths with the original variable names
            # The BWs returned are in the same order as X_ordered (i.e., permutation order)
            for j, name_in_order in enumerate(ordered_names):
                result[f'BW_{name_in_order}'] = bws[j]
            
            results_list.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{len(all_permutations)} permutations. Current BWs: {bws[:3]}...")
                
        except Exception as e:
            # Handle potential convergence errors during the search
            print(f"Error encountered for permutation {i+1} ({ordered_names}): {e}")
            # Append a placeholder result if an error occurs
            result = {'Order_ID': i + 1, 'Order_Sequence': str(ordered_names)}
            for name in base_names:
                result[f'BW_{name}'] = np.nan
            results_list.append(result)
            
    # Convert the list of results to a DataFrame for analysis
    results_df = pd.DataFrame(results_list)
    return results_df



# --- Main execution block ---
if __name__ == "__main__":
    
    # 1. Generate data
    coords, y_raw, X_raw, true_betas = generate_data(N=1000, seed=42)

    # 2. Prepare data for MGWR (Standardization)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    model_X = (X_raw - X_mean) / X_std
    y_mean = y_raw.mean(axis=0)
    y_std = y_raw.std(axis=0)
    model_y = (y_raw - y_mean) / y_std
    model_coords = coords

    # 3. Print Summary and Plot (if flag is set)
    if SUMMARY_DATA:
        print("Generating simulated data with 6 covariates...")
        print("\nData generation complete. Summary of True Betas:")
        print(true_betas.head())
        print(f"\nModel X shape: {model_X.shape}")
        print(f"Model y shape: {model_y.shape}")
        print("Data is standardized and ready for MGWR.")
        plot_betas(true_betas)

    # 4. Run the permutation test (WARNING: This will be time-consuming!)
    print("\n--- Running 6! (720) Permutation Test without Intercept ---")
    
    # You MUST uncomment the line below to run the full experiment:
    permutation_results = run_permutation_test(model_coords, model_y, model_X)

    # Since running 720 MGWR optimizations is computationally intensive, 
    # the function call is commented out for safety in this environment.
    # When running locally, uncomment the line above and ensure all necessary
    # libraries (mgwr, numpy, pandas) are installed.
    
    print("\n!!! ACTION REQUIRED: Uncomment the 'permutation_results' line to run the test locally. !!!")
    print("This experiment tests 720 different Backfitting sequences and will take time.")
    
    # After running, you can analyze the stability:
    print("\n--- Analysis Example (After running the test) ---")
    print(permutation_results.describe())
    print(f"Max Bandwidth Range for X1 (High Heterogeneity): {permutation_results['BW_X1'].max() - permutation_results['BW_X1'].min():.2f}")
    print(f"Max Bandwidth Range for X6 (Global): {permutation_results['BW_X6'].max() - permutation_results['BW_X6'].min():.2f}")

    filename = 'mgwr_x_order_permutation_results.csv'
    permutation_results.to_csv(filename, index=False)
    print(f"SAVEDSSSSSS~~~: {filename}")