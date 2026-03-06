"""
Thesis: Spatial Distribution and Uncertainty Assessment of Heavy Metal Contamination - Debrecen
AI-Assisted Reproducible Workflow
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
from thesis_workflow.core import ThesisWorkflow # Assuming logic is refactored here

# Constants
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'XRF_commonSpatial_Median.csv')
OUTPUT_DIR = r"d:\Debrecen-város(1)\Final_Thesis_Results"
METALS = [
    'Medián_As', 'Medián_Ca', 'Medián_Cd', 'Medián_Co', 'Medián_Cr', 
    'Medián_Cu', 'Medián_Fe', 'Medián_K', 'Medián_Mn', 'Medián_Mo', 
    'Medián_Ni', 'Medián_Pb', 'Medián_Ti', 'Medián_V', 'Medián_Zn'
]

def main():
    print("="*50)
    print("DEBRECEN SOIL CONTAMINATION: MASTER ANALYSIS WORKFLOW")
    print("="*50)
    
    # 1. Descriptive Statistics (Obj 1)
    print("\n[Phase 1] Running Descriptive Statistics...")
    # This logic is often handled within summarize_data.py or core.py
    import summarize_data
    summarize_data.run_summary()

    # 2. Spatial Interpolation & Performance (Obj 2)
    print("\n[Phase 2] Comparing Interpolation Models (IDW vs Kriging)...")
    import compare_interpolation
    compare_interpolation.run_comparison()

    # 3. Spatial Uncertainty & Probability (Obj 3)
    print("\n[Phase 3] Generating Risk & Probability Maps...")
    import generate_probability_maps
    generate_probability_maps.run_analysis()

    # 4. Land Use & Hotspot Assessment (Obj 4)
    print("\n[Phase 4] Analyzing Land Use Patterns & Spatial Hotspots...")
    import analyze_land_use
    import hotspot_analysis
    analyze_land_use.analyze_land_use()
    hotspot_analysis.run_hotspot_analysis()

    # 5. Multivariate Statistics (Obj 5)
    print("\n[Phase 5] Running Multivariate Statistics (PCA & Correlation)...")
    import run_multivariate_stats
    run_multivariate_stats.run_multivariate_analysis()

    # 6. Web GIS Platform (Obj 6)
    print("\n[Phase 6] Generating Web GIS Platform for dissemination...")
    import create_web_gis
    create_web_gis.generate_config()
    sync_assets()
    
    print("\n" + "="*50)
    print("SUCCESS: Full thesis workflow (Objectives 1-6) completed.")
    print(f"Results are stored in {OUTPUT_DIR}")
    print("Open 'web_dashboard/index.html' to view the interactive results.")
    print("="*50)

def sync_assets():
    """Sync generated plots to the web dashboard asset folder."""
    import shutil
    web_assets = r"d:\Debrecen-város(1)\web_dashboard\assets"
    source_dirs = [
        r"d:\Debrecen-város(1)\InterpolationPerformance",
        r"d:\Debrecen-város(1)\ProbabilityMaps",
        os.path.join(PROJECT_ROOT, 'results', 'LandUseAnalysis'),
        r"d:\Debrecen-város(1)\HotspotAnalysis",
        r"d:\Debrecen-város(1)\StatisticalAnalysis",
        r"d:\Debrecen-város(1)\TopoMetalMaps"
    ]
    
    if not os.path.exists(web_assets):
        os.makedirs(web_assets)
    
    print("Syncing assets to dashboard...")
    for s_dir in source_dirs:
        if os.path.exists(s_dir):
            for file in os.listdir(s_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.json')):
                    shutil.copy2(os.path.join(s_dir, file), os.path.join(web_assets, file))
    
    # Also sync sampling locations if it exists in root
    root_assets = [r"d:\Debrecen-város(1)\sampling_locations.png"]
    for asset in root_assets:
        if os.path.exists(asset):
            shutil.copy2(asset, os.path.join(web_assets, os.path.basename(asset)))

if __name__ == "__main__":
    main()
