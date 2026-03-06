import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configuration
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'XRF_commonSpatial_Median.csv')
OUTPUT_DIR = r"d:\Debrecen-város(1)\KrigingHeatmaps"
GRID_SIZE = (100, 200) # Higher resolution for final heatmaps

# Metals to interpolate
METALS = {
    'Medián_As': 'Arsenic (As)',
    'Medián_Pb': 'Lead (Pb)',
    'Medián_Zn': 'Zinc (Zn)',
    'Medián_Cu': 'Copper (Cu)',
    'Medián_Ni': 'Nickel (Ni)',
    'Medián_Cr': 'Chromium (Cr)'
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def ordinary_kriging(coords, values, target_coords, sill, range_val, nugget=0.1):
    """Ordinary Kriging implementation with Exponential variogram."""
    # Exponential covariance: C(h) = Sill * exp(-3h/Range)
    def cov_func(h):
        return sill * np.exp(-3 * h / range_val)

    n = len(coords)
    # Distances between data points
    d_data = cdist(coords, coords)
    C_data = cov_func(d_data)
    
    # Add nugget effect to diagonal
    C_data += np.eye(n) * nugget
    
    # Ordinary Kriging matrix (n+1 x n+1)
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = C_data
    K[n, :n] = 1.0
    K[:n, n] = 1.0
    
    # Distances between targets and data
    d_target = cdist(target_coords, coords)
    C_target = cov_func(d_target)
    
    # Target vectors for OK (n+1)
    # Each row is [C_target_i, 1]
    R = np.ones((target_coords.shape[0], n + 1))
    R[:, :n] = C_target
    
    # Solve K * w = R
    try:
        # Batch solve for all target points
        weights = np.linalg.solve(K, R.T).T
    except np.linalg.LinAlgError:
        print(" Warning: OK Matrix is singular, using pinv...")
        weights = np.linalg.pinv(K) @ R.T
        weights = weights.T
    
    # OK Estimate: Sum(w_i * z_i)
    # Note: weights are (n_targets, n+1), we only use the first n weights
    est = weights[:, :n] @ values
    return est

def process_metal(column, title):
    print(f"Processing {title}...")
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    
    # Drop rows with NaN in target column
    df = df.dropna(subset=[column])
    
    # Coordinates and Values
    x = df['EOVXX'].values
    y = df['EOVYY'].values
    z = df[column].values
    
    # Deduplicate
    coords = np.column_stack((x, y))
    _, uix = np.unique(coords.round(1), axis=0, return_index=True)
    coords = coords[uix]
    z = z[uix]
    
    # Area extent
    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()
    
    # Buffer for visualization
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    
    xi = np.linspace(x_min - pad_x, x_max + pad_x, GRID_SIZE[1])
    yi = np.linspace(y_min - pad_y, y_max + pad_y, GRID_SIZE[0])
    GX, GY = np.meshgrid(xi, yi)
    target_coords = np.column_stack((GX.ravel(), GY.ravel()))
    
    # Kriging Parameters (Simplified default)
    sill = np.var(z)
    range_val = 6000 # 6km spatial correlation distance
    
    print(f" Interpolating {len(target_coords)} points...")
    # Chunking to avoid memory issues with solve
    chunk_size = 1000
    est_full = []
    for i in range(0, len(target_coords), chunk_size):
        chunk = target_coords[i:i+chunk_size]
        est_chunk = ordinary_kriging(coords, z, chunk, sill, range_val)
        est_full.append(est_chunk)
    
    est = np.concatenate(est_full).reshape(GRID_SIZE)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    v_min = np.percentile(z, 5)
    v_max = np.percentile(z, 95)
    
    im = plt.imshow(est, extent=(xi.min(), xi.max(), yi.min(), yi.max()), 
                    origin='lower', cmap='YlOrRd', vmin=v_min, vmax=v_max)
    plt.colorbar(im, label='Concentration (ppm)')
    plt.scatter(coords[:,0], coords[:,1], c='black', s=10, alpha=0.4, label='Samples')
    
    plt.title(f"Ordinary Kriging Heatmap: {title} in Debrecen", fontsize=16)
    plt.xlabel("EOV Easting (m)")
    plt.ylabel("EOV Northing (m)")
    plt.legend()
    
    out_name = f"{title.replace(' ', '_').replace(')', '').replace('(', '')}_Kriging.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" Saved: {out_path}")

if __name__ == "__main__":
    for col, title in METALS.items():
        try:
            process_metal(col, title)
        except Exception as e:
            print(f"Error processing {title}: {e}")
