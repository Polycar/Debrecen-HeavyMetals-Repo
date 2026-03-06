# Debrecen Heavy Metals Spatial Analysis & WebGIS Dashboard

This repository contains the data, geostatistical analysis scripts, and an interactive WebGIS dashboard for analyzing soil heavy metal contamination in the city of Debrecen, Hungary. 

## Repository Structure

- `data/`: Contains the primary XRF sample data and geographical shapefiles.
- `scripts/`: Python scripts for computing experimental variograms, Sequential Gaussian Simulation (SGS), mapping, and multivariate statistical modeling.
- `dashboard/`: A full-stack Streamlit and Flask application for interactive spatial visualization of risk areas, hotspots, and probabilistic interpolation surfaces.
- `results/`: Cached results of expensive spatial computations (such as IDW, Kriging, Land Use intersections, and simulated uncertainty maps).
- `thesis/`: Contains the academic writeup of this spatial analysis (e.g., Chapter 4).

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Debrecen-HeavyMetals-Repo
   ```

2. **Install dependencies:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

## Running the WebGIS Dashboard

The interactive WebGIS dashboard allows stakeholders to explore heavy metal distributions, scientific summaries, and cumulative risk mappings seamlessly.

**To run the frontend dashboard:**
```bash
cd dashboard
python -m streamlit run streamlit_dashboard.py
```

*Note: For the Flask-based WebGIS routing, you can run `python app.py`.*

## Running the Geostatistical Scripts

All paths in the scripts dynamically resolve to the repository root. You can run any script from the `scripts/` folder safely.

```bash
cd scripts
python calculate_all_thesis_stats.py
python run_sgsim.py
```

## Methodology

The spatial modeling utilizes:
- **Sequential Gaussian Simulation (SGS)** for local spatial uncertainty.
- **Ordinary Kriging (OK)** and **Simple Kriging (SK)** for deterministic surface modeling.
- **Hazard Quotient (HQ)** thresholding against standard Hungarian regulatory limits for soils. 
