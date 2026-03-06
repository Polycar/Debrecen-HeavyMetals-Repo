import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configuration
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'XRF_commonSpatial_Median.csv')
OUTPUT_DIR = r"d:\Debrecen-város(1)\HotspotAnalysis"
METALS = [
    'Medián_As', 'Medián_Ca', 'Medián_Cd', 'Medián_Co', 'Medián_Cr', 
    'Medián_Cu', 'Medián_Fe', 'Medián_K', 'Medián_Mn', 'Medián_Mo', 
    'Medián_Ni', 'Medián_Pb', 'Medián_Ti', 'Medián_V', 'Medián_Zn'
]
DISTANCE_THRESHOLD = 500  # 500 meters for local neighborhood

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def getis_ord_gi_star(coords, values, distance_threshold):
    """
    Custom implementation of Getis-Ord Gi* local statistic.
    Gi* = (Sum(w_ij * x_j) - X_bar * Sum(w_ij)) / (S * sqrt((n * Sum(w_ij^2) - (Sum(w_ij))^2) / (n - 1)))
    """
    n = len(values)
    x_bar = np.mean(values)
    S = np.sqrt(np.sum(values**2) / n - x_bar**2)
    
    # Spatial weights matrix (binary distance band)
    dists = cdist(coords, coords)
    W = (dists <= distance_threshold).astype(float)
    
    wi_sum = W.sum(axis=1)
    wi2_sum = (W**2).sum(axis=1)
    
    local_sum = (W * values).sum(axis=1)
    
    numerator = local_sum - x_bar * wi_sum
    denominator = S * np.sqrt((n * wi2_sum - wi_sum**2) / (n - 1))
    
    # Avoid division by zero
    gi_star = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return gi_star

def run_hotspot_analysis():
    df = pd.read_csv(CSV_FILE, encoding='latin1').dropna(subset=METALS)
    coords = df[['EOVXX', 'EOVYY']].values
    
    for metal in METALS:
        print(f"Analyzing Hotspots for {metal}...")
        values = df[metal].values
        
        gi_values = getis_ord_gi_star(coords, values, DISTANCE_THRESHOLD)
        
        # Plot
        plt.figure(figsize=(10, 12))
        # Color by Gi* values (Cold to Hot)
        # Gi* > 1.96 = Hotspot 95% confidence
        # Gi* < -1.96 = Coldspot 95% confidence
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=gi_values, cmap='RdYlBu_r', s=20, edgecolor='white', linewidth=0.1, vmin=-3, vmax=3)
        plt.colorbar(sc, label='Getis-Ord Gi* Z-score')
        
        plt.title(f"Objective 4: Spatial Hotspots for {metal.replace('Medián_', '')}\n(Distance Threshold: {DISTANCE_THRESHOLD}m)", fontsize=14)
        plt.xlabel("EOV Easting (m)")
        plt.ylabel("EOV Northing (m)")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        out_name = f"{metal.replace('Medián_', '')}_Hotspots.png"
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=200, bbox_inches='tight')
        plt.close()
        
        # Save Z-scores back to CSV for later use
        df[f'{metal}_Gi_Zscore'] = gi_values

    df.to_csv(os.path.join(OUTPUT_DIR, "Hotspot_Results.csv"), index=False)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_hotspot_analysis()
