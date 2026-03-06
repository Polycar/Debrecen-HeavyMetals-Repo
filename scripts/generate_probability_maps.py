import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configuration
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'XRF_commonSpatial_Median.csv')
OUTPUT_DIR = r"d:\Debrecen-város(1)\ProbabilityMaps"
GRID_SIZE = (60, 100) # Lower resolution for faster SGsim (can increase for final)
NUM_REALIZATIONS = 30 # Number of realizations for probability estimation

# Regulatory Thresholds (mg/kg or ppm) - Hungarian 6/2009. (IV. 14.) Joint Decree
THRESHOLDS = {
    'Medián_Pb': 100,
    'Medián_Zn': 200,
    'Medián_Cu': 75,
    'Medián_As': 15,
    'Medián_Ni': 40,
    'Medián_Cr': 75,
    'Medián_Cd': 1
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def normal_score_transform(z):
    """Transform data to normal distribution."""
    rank = z.argsort().argsort()
    return norm.ppf((rank + 1) / (len(z) + 1))

def simple_kriging(coords, values, target_coords, sill, range_val, nugget=0.1):
    """Simple Kriging with known mean 0 (for Normal Scores)."""
    # Exponential covariance
    def cov_func(h):
        return sill * np.exp(-3 * h / range_val)

    n = len(coords)
    d_data = cdist(coords, coords)
    C_data = cov_func(d_data) + np.eye(n) * nugget
    
    d_target = cdist(target_coords, coords)
    C_target = cov_func(d_target)
    
    # Solve C_data * w = C_target
    try:
        weights = np.linalg.solve(C_data, C_target.T).T
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(C_data) @ C_target.T
        weights = weights.T
    
    # Mean is 0 for normal scores
    est = weights @ values
    # Variance = Sill - weights * C_target
    var = sill - (weights * C_target).sum(axis=1)
    return est, np.clip(var, 0, sill)

def run_sgsim_probability(column, threshold_val, metal_name):
    print(f"--- Calculating Probability Maps for {metal_name} ---")
    df = pd.read_csv(CSV_FILE, encoding='latin1').dropna(subset=[column])
    z_raw = df[column].values
    coords_raw = df[['EOVXX', 'EOVYY']].values
    
    # Transform
    z_norm = normal_score_transform(z_raw)
    
    # Extent
    x_min, x_max = coords_raw[:, 0].min(), coords_raw[:, 0].max()
    y_min, y_max = coords_raw[:, 1].min(), coords_raw[:, 1].max()
    xi = np.linspace(x_min, x_max, GRID_SIZE[1])
    yi = np.linspace(y_min, y_max, GRID_SIZE[0])
    GX, GY = np.meshgrid(xi, yi)
    grid_coords = np.column_stack((GX.ravel(), GY.ravel()))
    
    # Semi-variogram params
    sill = 1.0 # Normal score sill is 1.0
    range_val = 5000 
    
    # Storage for realizations
    exceedance_count = np.zeros(len(grid_coords))
    all_realizations = []
    
    for r in range(NUM_REALIZATIONS):
        if (r+1) % 5 == 0: print(f"  Simulating realization {r+1}/{NUM_REALIZATIONS}...")
        
        # Start with original data
        sim_coords = coords_raw.copy()
        sim_values = z_norm.copy()
        
        # Random path
        indices = np.random.permutation(len(grid_coords))
        realization = np.zeros(len(grid_coords))
        
        for idx in indices:
            target = grid_coords[idx:idx+1]
            # Kriging estimate using current "known" points (data + simulated)
            # Optimization: limit to nearest 15 points
            dists = cdist(target, sim_coords)[0]
            nearest = np.argsort(dists)[:15]
            
            k_mean, k_var = simple_kriging(sim_coords[nearest], sim_values[nearest], target, sill, range_val)
            
            # Simulate value
            val = np.random.normal(k_mean[0], np.sqrt(k_var[0]))
            realization[idx] = val
            
            # Update known points
            sim_coords = np.vstack([sim_coords, target])
            sim_values = np.append(sim_values, val)
        
        # Back-transform realization
        # Mapping realization (Normal Score) -> Percentile -> Empirical CDF
        rank_real = realization.argsort().argsort()
        percentiles = (rank_real + 1) / (len(realization) + 1)
        z_back = np.percentile(z_raw, percentiles * 100)
        
        # Count exceedance
        exceedance_count += (z_back > threshold_val)
        all_realizations.append(z_back)

    # Calculate Probability & Uncertainty
    stack = np.array(all_realizations) # (NUM_REALIZATIONS, num_pixels)
    prob = (exceedance_count / NUM_REALIZATIONS).reshape(GRID_SIZE)
    uncertainty = np.std(stack, axis=0).reshape(GRID_SIZE)
    
    # 1. Plot Probability
    plt.figure(figsize=(12, 10))
    im = plt.imshow(prob, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(im, label=f'Probability of Exceeding {threshold_val} ppm')
    plt.scatter(coords_raw[:,0], coords_raw[:,1], c='black', s=5, alpha=0.3)
    plt.title(f"Objective 3: Risk Map for {metal_name} (Threshold: {threshold_val} ppm)\nSequential Gaussian Simulation (n={NUM_REALIZATIONS})", fontsize=14)
    plt.xlabel("EOV Easting (m)")
    plt.ylabel("EOV Northing (m)")
    out_path_prob = os.path.join(OUTPUT_DIR, f"{metal_name}_Probability.png")
    plt.savefig(out_path_prob, dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Plot Uncertainty (Standard Deviation)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(uncertainty, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='magma')
    plt.colorbar(im, label='Standard Deviation (Uncertainty)')
    plt.scatter(coords_raw[:,0], coords_raw[:,1], c='white', s=5, alpha=0.3)
    plt.title(f"Objective 3: Spatial Uncertainty for {metal_name}\nLocal Standard Deviation of {NUM_REALIZATIONS} Realizations", fontsize=14)
    plt.xlabel("EOV Easting (m)")
    plt.ylabel("EOV Northing (m)")
    out_path_var = os.path.join(OUTPUT_DIR, f"{metal_name}_Uncertainty.png")
    plt.savefig(out_path_var, dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Save Sample Realizations (1, 2, 3)
    for i in range(min(3, NUM_REALIZATIONS)):
        real_img = all_realizations[i].reshape(GRID_SIZE)
        plt.figure(figsize=(10, 8))
        plt.imshow(real_img, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='Spectral_r')
        plt.title(f"SGsim Realization #{i+1} for {metal_name}")
        plt.colorbar(label='Concentration (ppm)')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metal_name}_Realization_{i+1}.png"), dpi=150)
        plt.close()

    print(f" Saved probability, uncertainty, and realizations for {metal_name}")

def run_analysis():
    print("="*50)
    print("STARTING OBJECTIVE 3 REWORK: ADVANCED RISK & UNCERTAINTY")
    print("="*50)
    # Process all metals that have defined regulatory thresholds
    for metal_col, threshold in THRESHOLDS.items():
        metal_name = metal_col.replace('Medián_', '')
        run_sgsim_probability(metal_col, threshold, metal_name)

if __name__ == "__main__":
    run_analysis()
