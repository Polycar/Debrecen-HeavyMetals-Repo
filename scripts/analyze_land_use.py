import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configuration
GDB_FILE = r"d:\Debrecen-város(1)\Debrecen-város\DebrecenVáros\XRF_geostati.gdb"
LAYER_NAME = "Toco_pontok_minden_urbanatlas"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'LandUseAnalysis')
METALS = [
    'Medián_As', 'Medián_Ca', 'Medián_Cd', 'Medián_Co', 'Medián_Cr', 
    'Medián_Cu', 'Medián_Fe', 'Medián_K', 'Medián_Mn', 'Medián_Mo', 
    'Medián_Ni', 'Medián_Pb', 'Medián_Ti', 'Medián_V', 'Medián_Zn'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def analyze_land_use():
    print(f"Reading Layer: {LAYER_NAME}...")
    df = gpd.read_file(GDB_FILE, layer=LAYER_NAME)
    
    # Filter for relevant columns
    cols = METALS + ['class_2018', 'Code_18']
    data = df[cols].copy()
    
    # Drop rows with missing land use or metal data
    data = data.dropna(subset=['class_2018'] + METALS)
    
    # Clean up class names (remove codes if present)
    data['class_name'] = data['class_2018'].str.replace(r'^\d+\s+', '', regex=True)
    
    # 1. Bar plot of Sample counts per Land Use
    plt.figure(figsize=(12, 6))
    sns.countplot(y='class_name', data=data, palette='viridis', order=data['class_name'].value_counts().index)
    plt.title("Sample Distribution by Urban Atlas Land Use Class", fontsize=15)
    plt.xlabel("Number of Samples")
    plt.ylabel("Land Use Class (2018)")
    plt.savefig(os.path.join(OUTPUT_DIR, "Sample_Distribution_LandUse.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Box plots for each metal
    for metal in METALS:
        plt.figure(figsize=(14, 8))
        # Use log scale for better visualization if wide range
        sns.boxplot(x=metal, y='class_name', data=data, palette='YlOrRd', showfliers=False)
        sns.stripplot(x=metal, y='class_name', data=data, color='black', size=3, alpha=0.3)
        
        plt.title(f"Objective 4: {metal.replace('Medián_', '')} Concentration by Land Use Class", fontsize=16)
        plt.xlabel(f"{metal.replace('Medián_', '')} Concentration (ppm)")
        plt.ylabel("Land Use Class")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        out_name = f"{metal.replace('Medián_', '')}_by_LandUse.png"
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=200, bbox_inches='tight')
        plt.close()
        print(f" Saved: {out_name}")

    # 3. Summary Statistics Table
    summary = data.groupby('class_name')[METALS].agg(['mean', 'median', 'std', 'count'])
    summary_path = os.path.join(OUTPUT_DIR, "LandUse_Metal_Summary.csv")
    try:
        summary.to_csv(summary_path)
        print(f"Saved Summary Table to {summary_path}")
    except PermissionError:
        import time
        alt_path = os.path.join(OUTPUT_DIR, f"LandUse_Metal_Summary_{int(time.time())}.csv")
        summary.to_csv(alt_path)
        print(f"Permission denied for {summary_path}. Saved to {alt_path} instead. Please close the file if it is open in Excel.")

if __name__ == "__main__":
    analyze_land_use()
