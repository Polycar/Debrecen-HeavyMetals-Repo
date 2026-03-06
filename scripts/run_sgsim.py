import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata
from scipy.spatial.distance import cdist
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configuration
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'XRF_commonSpatial_Median.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'SGsimResults')
GRID_SIZE = (40, 80)    # Coarser grid for speed
REALIZATIONS = 5        # Number of simulations per metal
RANGE_VAL = 6000        # 6km spatial correlation distance

# Metals to simulate
METALS = {
    'Medián_As': 'Arsenic (As)',
    'Medián_Pb': 'Lead (Pb)',
    'Medián_Zn': 'Zinc (Zn)',
    'Medián_Cu': 'Copper (Cu)',
    'Medián_Ni': 'Nickel (Ni)',
    'Medián_Cr': 'Chromium (Cr)',
    'Medián_Cd': 'Cadmium (Cd)'
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def normal_score_transform(data):
    """Transform data to standard normal distribution."""
    n = len(data)
    ranks = rankdata(data)
    # Avoid 0 and 1 for ppf
    probs = (ranks - 0.5) / n
    nscore = norm.ppf(probs)
    return nscore, data

def back_transform(sim_data, original_data):
    """Transform simulated normal scores back to original distribution."""
    # Ensure no NaNs from simulation
    sim_data = np.nan_to_num(sim_data, nan=0.0)
    
    # Sort original for lookup
    sorted_orig = np.sort(original_data)
    
    # Get percentiles of normal scores
    probs = norm.cdf(sim_data)
    
    # Map to original values with clipping
    indices = (probs * (len(sorted_orig) - 1))
    indices = np.clip(indices, 0, len(sorted_orig) - 1).astype(int)
    return sorted_orig[indices]

def simple_kriging(coords, values, target_coords, sill, range_val):
    """Very simple SK implementation with Exponential variogram."""
    # Exponential covariance: C(h) = Sill * exp(-3h/Range)
    def cov_func(h):
        return sill * np.exp(-3 * h / range_val)

    # Distances between data points
    d_data = cdist(coords, coords)
    C_data = cov_func(d_data)
    
    # Distances between targets and data
    d_target = cdist(target_coords, coords)
    C_target = cov_func(d_target)
    
    # Robust inversion using SVD if solve fails
    try:
        weights = np.linalg.solve(C_data + np.eye(len(coords)) * 1e-4, C_target.T).T
    except np.linalg.LinAlgError:
        print(" Using pinv for stability...")
        weights = np.linalg.pinv(C_data) @ C_target.T
        weights = weights.T
    
    est = weights @ values
    var = sill - np.sum(weights * C_target, axis=1)
    return est, np.maximum(var, 0)

def run_simulation(metal_col, metal_label):
    print(f"--- Starting SGsim for {metal_label} ---")
    
    # Create metal-specific directory
    metal_dir = os.path.join(OUTPUT_DIR, metal_label.replace(' ', '_').replace(')', '').replace('(', ''))
    if not os.path.exists(metal_dir):
        os.makedirs(metal_dir)

    df = pd.read_csv(CSV_FILE, encoding='latin1')
    
    # Drop rows with NaN in the target metal
    df = df.dropna(subset=[metal_col])
    print(f"Using {len(df)} valid data points for {metal_col}")
    
    # Coordinates and Values
    x_orig = df['EOVXX'].values
    y_orig = df['EOVYY'].values
    z_orig = df[metal_col].values
    
    # Deduplicate
    coords_orig = np.column_stack((x_orig, y_orig))
    _, uix = np.unique(coords_orig.round(1), axis=0, return_index=True)
    coords_orig = coords_orig[uix]
    z_orig = z_orig[uix]
    print(f"Using {len(z_orig)} unique locations.")
    
    # Phase 1: Transform
    print("Performing Normal Score Transform...")
    z_nscore, _ = normal_score_transform(z_orig)
    
    # Initial Variogram Parameters
    sill = 1.0        # nscore has unit variance
    
    # Phase 2: Simulation Grid
    x_min, x_max = coords_orig[:,0].min(), coords_orig[:,0].max()
    y_min, y_max = coords_orig[:,1].min(), coords_orig[:,1].max()
    
    grid_x = np.linspace(x_min, x_max, GRID_SIZE[1])
    grid_y = np.linspace(y_min, y_max, GRID_SIZE[0])
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_coords = np.column_stack((gx.ravel(), gy.ravel()))
    
    # Grid sequence
    indices = np.arange(len(grid_coords))
    
    # Define color scale for consistency across realizations
    v_min = np.percentile(z_orig, 1)
    v_max = np.percentile(z_orig, 99)
    
    for r in range(REALIZATIONS):
        print(f"Generating Realization {r+1}/{REALIZATIONS}...")
        np.random.shuffle(indices)
        
        sim_values = np.zeros(len(grid_coords))
        known_coords = coords_orig.copy()
        known_values = z_nscore.copy()
        
        # We simulate in chunks for speed
        chunk_size = 200
        for i in range(0, len(grid_coords), chunk_size):
            chunk_idx = indices[i:i+chunk_size]
            target = grid_coords[chunk_idx]
            
            # Simple Kriging estimate and variance
            est, var = simple_kriging(known_coords, known_values, target, sill, RANGE_VAL)
            
            # Draw from N(est, var)
            draws = np.random.normal(est, np.sqrt(var))
            sim_values[chunk_idx] = draws

        # Back transform
        print(" Back-transforming...")
        z_sim_orig = back_transform(sim_values, z_orig)
        z_sim_grid = z_sim_orig.reshape(GRID_SIZE)
        
        # Plot
        plt.figure(figsize=(10, 8))
        im = plt.imshow(z_sim_grid, extent=(x_min, x_max, y_min, y_max), 
                        origin='lower', cmap='YlOrRd', vmin=v_min, vmax=v_max)
        plt.colorbar(im, label=f'{metal_label} (ppm)')
        plt.title(f"SGsim: {metal_label} - Realization {r+1}")
        plt.scatter(coords_orig[:,0], coords_orig[:,1], c='black', s=5, alpha=0.5, label='Samples')
        plt.xlabel('EOV Easting')
        plt.ylabel('EOV Northing')
        
        out_path = os.path.join(metal_dir, f"Realization_{r+1}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {out_path}")

if __name__ == "__main__":
    for col, label in METALS.items():
        try:
            run_simulation(col, label)
        except Exception as e:
            print(f"Error during simulation for {label}: {e}")
            import traceback
            traceback.print_exc()
