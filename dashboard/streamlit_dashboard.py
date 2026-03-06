"""
Debrecen Heavy Metal WebGIS Dashboard
======================================
Streamlit + Folium + Plotly dashboard for soil contamination analysis.
Features:
  - Interactive Folium map with raster overlays (Kriging, Probability, Uncertainty, SGS)
  - Snowflake / Radar charts for multi-element fingerprinting
  - Sample point explorer with popups
  - Dark-themed, premium UI

Run with:  python -m streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import base64
import folium
import folium.plugins
import geopandas as gpd
import plotly.graph_objects as go
from streamlit_folium import st_folium
from pyproj import Transformer

# ──────────────────────────── CONFIG ────────────────────────────
st.set_page_config(
    page_title="Debrecen WebGIS — Heavy Metal Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
ASSETS_DIR = os.path.join(BASE_DIR, "web_dashboard", "assets")
DATA_PATH = os.path.join(REPO_ROOT, "data", "XRF_commonSpatial_Median.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "web_dashboard", "layers_config.json")
DISTRICTS_PATH = os.path.join(REPO_ROOT, "data", "nehezfem-zonal.shp")

METALS_INFO = {
    "As": {"name": "Arsenic",   "limit": 15,   "color": "#ef4444", "unit": "mg/kg"},
    "Cd": {"name": "Cadmium",   "limit": 1,    "color": "#f97316", "unit": "mg/kg"},
    "Cr": {"name": "Chromium",  "limit": 75,   "color": "#eab308", "unit": "mg/kg"},
    "Cu": {"name": "Copper",    "limit": 75,   "color": "#22c55e", "unit": "mg/kg"},
    "Ni": {"name": "Nickel",    "limit": 40,   "color": "#06b6d4", "unit": "mg/kg"},
    "Pb": {"name": "Lead",      "limit": 100,  "color": "#8b5cf6", "unit": "mg/kg"},
    "Zn": {"name": "Zinc",      "limit": 200,  "color": "#ec4899", "unit": "mg/kg"},
    # Secondary metals found in CSV
    "Fe": {"name": "Iron",      "limit": 30000,"color": "#94a3b8", "unit": "mg/kg"},
    "Mn": {"name": "Manganese", "limit": 1000, "color": "#64748b", "unit": "mg/kg"},
    "Ti": {"name": "Titanium",  "limit": 5000, "color": "#475569", "unit": "mg/kg"},
    "V":  {"name": "Vanadium",  "limit": 100,  "color": "#334155", "unit": "mg/kg"},
}

# Threshold Alert Levels (Hazard Quotient)
HQ_LEVELS = {
    "safe":    {"label": "Safe",      "color": "#22c55e", "icon": "✓"},
    "caution": {"label": "Caution",   "color": "#eab308", "icon": "●"},
    "warning": {"label": "Warning",   "color": "#f97316", "icon": "⚡"},
    "danger":  {"label": "Hazardous", "color": "#ef4444", "icon": "⚠️"},
}

LAYER_TYPES = {
    "smooth_heatmap": "✨ Smooth Heatmap (Live)",
    "health_risk":   "🩺 Health Risk (HQ)",
    "kriging":       "Ordinary Kriging",
    "probability":   "Probability of Exceedance",
    "uncertainty":   "Spatial Uncertainty (StdDev)",
    "realization":   "SGS Realization",
    "hotspot":       "Hotspot Analysis",
}


# ──────────────────────────── STYLES ────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMetric label, .main-title {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Premium Animated Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at 80% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.05) 0%, transparent 40%);
        pointer-events: none;
        z-index: -1;
    }

    /* Glassmorphism Tokens */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }

    /* Branded Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px 40px;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-text h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Scientific Stats Ribbon */
    .stats-ribbon {
        display: flex;
        gap: 16px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stat-item {
        flex: 1;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-item:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(16, 185, 129, 0.3);
        transform: translateY(-2px);
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-header {
        padding: 20px 0;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }

    /* Portfolio Enhancement */
    .portfolio-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .portfolio-card:hover {
        transform: scale(1.02);
        border-color: #10b981;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    /* Custom Floating Legend */
    .floating-legend {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
    }

    /* Inputs & Selectors */
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-radius: 8px !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.4);
        padding: 4px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #10b981 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

    .block-container { padding-top: 2rem !important; }
    iframe { border-radius: 16px !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────── DATA ────────────────────────────
def load_data():
    """Load and prepare the XRF sampling data."""
    df = pd.read_csv(DATA_PATH, encoding='latin1')
    # Rename columns to clean names
    rename_map = {}
    for col in df.columns:
        # Match "Medi" followed by the metal ID (e.g., "Medin_Pb")
        for metal_id in METALS_INFO:
            if metal_id in col and 'Medi' in col:
                rename_map[col] = metal_id
    df = df.rename(columns=rename_map)
    
    # Clean numeric columns
    for m in METALS_INFO:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors='coerce')

    # Convert EOV to WGS84
    transformer = Transformer.from_crs("EPSG:23700", "EPSG:4326", always_xy=True)
    lngs, lats = transformer.transform(df['EOVXX'].values, df['EOVYY'].values)
    df['lat'] = lats
    df['lng'] = lngs
    return df

def create_correlation_matrix(df):
    """Create a Pearson correlation heatmap for all metals."""
    metals = list(METALS_INFO.keys())
    corr = df[metals].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=[METALS_INFO[m]['name'] for m in metals],
        y=[METALS_INFO[m]['name'] for m in metals],
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate='Metal 1: %{x}<br>Metal 2: %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="🧬 Metal Correlation Matrix (Pearson)", x=0.5, font=dict(family='Outfit', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    return fig

def create_landuse_boxplots(df, districts_gdf):
    """Create boxplots of metal concentrations per district type."""
    if districts_gdf is None: return None
    
    # Use SampleID or coordinates to join with districts
    # For now, we'll assume the districts shapefile already has summarized stats 
    # but a real thesis often needs the raw sample-to-district join.
    # We can use the 'tipus' from the shapefile if we join points to polygons.
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Convert df to GDF
    geometry = [Point(xy) for xy in zip(df.lng, df.lat)]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Spatial join
    joined = gpd.sjoin(points_gdf, districts_gdf[['tipus', 'geometry']], how="left", predicate="within")
    
    # We'll plot HQ (normalized) to make comparisons possible on one scale
    plot_df = joined.copy()
    for m in METALS_INFO:
        plot_df[f"{m}_HQ"] = plot_df[m] / METALS_INFO[m]['limit']
    
    # Melt for plotly
    melted = plot_df.melt(id_vars=['tipus'], value_vars=[f"{m}_HQ" for m in METALS_INFO], 
                          var_name='Metal', value_name='HQ')
    melted['Metal'] = melted['Metal'].str.replace('_HQ', '')
    melted['tipus'] = melted['tipus'].fillna('Unknown').apply(lambda x: x.replace('_', ' ').title())

    fig = go.Figure()
    for m in METALS_INFO:
        m_data = melted[melted['Metal'] == m]
        fig.add_trace(go.Box(
            y=m_data['HQ'],
            x=m_data['tipus'],
            name=METALS_INFO[m]['name'],
            marker_color=METALS_INFO[m]['color'],
            boxpoints=False
        ))
        
    fig.update_layout(
        title=dict(text="🏘️ Contamination Trends by Land-Use Type", x=0.5, font=dict(family='Outfit', size=16)),
        yaxis=dict(title="Hazard Quotient (x Limit)", gridcolor='rgba(255,255,255,0.05)', type='log'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.3)',
        font=dict(color='#f8fafc'),
        height=500,
        boxmode='group',
        margin=dict(t=60, b=80)
    )
    return fig


@st.cache_data
def load_config():
    """Load the layers config for map bounds."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


@st.cache_data
def load_districts():
    """Load district boundaries from the nehezfem-zonal shapefile."""
    if not os.path.exists(DISTRICTS_PATH):
        return None
    gdf = gpd.read_file(DISTRICTS_PATH)
    # Reproject EOV (EPSG:23700) to WGS84 (EPSG:4326)
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def get_image_base64(path):
    """Encode image to base64 for Folium overlay."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_asset_path(metal_id, layer_type):
    """Resolve asset filename for a given metal and layer type."""
    metal_name = METALS_INFO[metal_id]["name"]
    if layer_type == "kriging":
        return os.path.join(ASSETS_DIR, f"{metal_name}_{metal_id}_Topo.png")
    elif layer_type == "probability":
        return os.path.join(ASSETS_DIR, f"{metal_id}_Probability.png")
    elif layer_type == "uncertainty":
        return os.path.join(ASSETS_DIR, f"{metal_id}_Uncertainty.png")
    elif layer_type == "realization":
        return os.path.join(ASSETS_DIR, f"{metal_id}_Realization_1.png")
    elif layer_type == "hotspot":
        return os.path.join(ASSETS_DIR, f"{metal_id}_Hotspots.png")
    return None


# ──────────────────────────── CHARTS ────────────────────────────
def create_parallel_coordinates(df):
    """Create a parallel coordinates plot for multi-metal patterns."""
    metals = list(METALS_INFO.keys())
    
    # Normalize for better visualization
    df_norm = df.copy()
    for m in metals:
        df_norm[m] = df_norm[m] / METALS_INFO[m]['limit']
    
    # Color by Pb (example) or just one metal
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df_norm['Pb'],
                       colorscale = 'Viridis',
                       showscale = True,
                       reversescale = True,
                       cmin = 0, cmax = 2),
            dimensions = list([
                dict(range = [0, 2.5],
                     label = f"{METALS_INFO[m]['name']}<br>(xLimit)", values = df_norm[m])
                for m in metals
            ]),
            unselected = dict(line = dict(color = 'rgba(100,100,100,0.05)'))
        )
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=10),
        margin=dict(t=50, b=30, l=60, r=60),
        height=400,
        title=dict(
            text="🧬 Multi-Metal Signature Profile (Sample Population)",
            font=dict(size=14, family='Outfit'),
            x=0.5
        )
    )
    return fig

def create_population_heatmap(df):
    """Create a heatmap showing HQ (Hazard Quotient) across all metals for all samples."""
    metals = list(METALS_INFO.keys())
    
    # Calculate HQ for top 50 samples by total risk for better visibility
    df_hq = df.copy()
    for m in metals:
        df_hq[m] = df_hq[m] / METALS_INFO[m]['limit']
    
    df_hq['total_risk'] = df_hq[metals].sum(axis=1)
    top_samples = df_hq.sort_values('total_risk', ascending=False).head(40)
    
    fig = go.Figure(data=go.Heatmap(
        z=top_samples[metals].values,
        x=[METALS_INFO[m]['name'] for m in metals],
        y=top_samples['SampleID'],
        colorscale='RdYlGn_r',
        zmin=0, zmax=2,
        colorbar=dict(title=dict(text="Hazard Quotient (xLimit)", side="right")),
        hovertemplate='Sample: %{y}<br>Metal: %{x}<br>HQ: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=11),
        margin=dict(t=50, b=30, l=100, r=60),
        height=600,
        title=dict(
            text="🔥 Contamination Heatmap (Top 40 Highest-Risk Samples)",
            font=dict(size=14, family='Outfit'),
            x=0.5
        ),
        yaxis=dict(autorange="reversed")
    )
    return fig

def create_sample_risk_bars(df, sample_idx):
    """Create a modern bar-gauge for a single sample."""
    row = df.iloc[sample_idx]
    metals = list(METALS_INFO.keys())
    
    values = []
    colors = []
    for m in metals:
        val = row.get(m, 0)
        limit = METALS_INFO[m]['limit']
        hq = val / limit if val and limit else 0
        values.append(hq)
        
        if hq > 1.5: color = '#ef4444'
        elif hq > 1.0: color = '#f97316'
        elif hq > 0.75: color = '#eab308'
        else: color = '#22c55e'
        colors.append(color)

    fig = go.Figure()
    
    # Add safe zone background
    fig.add_vrect(x0=0, x1=1, fillcolor="#22c55e", opacity=0.05, layer="below", line_width=0)
    
    fig.add_trace(go.Bar(
        y=[METALS_INFO[m]['name'] for m in metals],
        x=values,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{v:.2f}x" for v in values],
        textposition='auto',
        hovertemplate='%{y}<br>Ratio: %{x:.2f}x limit<extra></extra>'
    ))
    
    # Threshold line
    fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444", line_width=2, 
                  annotation_text="Regulatory Limit", annotation_position="top right")

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.3)',
        font=dict(color='#f8fafc', size=12),
        margin=dict(t=50, b=30, l=120, r=40),
        height=450,
        xaxis=dict(title="Hazard Quotient (Concentration / Limit)", gridcolor='rgba(255,255,255,0.05)', range=[0, max(2, max(values)*1.1)]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', autorange="reversed"),
        title=dict(
            text=f"📊 Risk Profile: Sample {row['SampleID']}",
            font=dict(size=14, family='Outfit'),
            x=0.5
        )
    )
    return fig



def create_bar_comparison(df):
    """Create a grouped bar chart comparing mean values vs regulatory thresholds."""
    metals = list(METALS_INFO.keys())
    means = [df[m].mean() for m in metals]
    limits = [METALS_INFO[m]["limit"] for m in metals]
    names = [METALS_INFO[m]["name"] for m in metals]
    colors = [METALS_INFO[m]["color"] for m in metals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=means,
        name='Mean Concentration',
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{v:.1f}" for v in means],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=10),
    ))
    fig.add_trace(go.Scatter(
        x=names, y=limits,
        mode='markers+lines',
        name='Regulatory Limit',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(color='#ef4444', size=8, symbol='diamond'),
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='', tickfont=dict(color='#cbd5e1', size=12)),
        yaxis=dict(
            title=dict(text='Concentration (mg/kg)', font=dict(color='#94a3b8')),
            tickfont=dict(color='#94a3b8'),
            gridcolor='rgba(148,163,184,0.08)',
            type='log',
        ),
        legend=dict(font=dict(color='#cbd5e1'), bgcolor='rgba(30,41,59,0.8)'),
        margin=dict(t=20, b=40),
        height=350,
        barmode='group',
    )
    return fig


# ──────────────────────────── FOLIUM MAP ────────────────────────────
def create_folium_map(df, config, metal_id, layer_type, show_points, opacity,
                      show_districts=False, districts_gdf=None, cumulative_risk=False):
    """Build an interactive Folium map with raster overlay and optional sample points."""
    bounds = config["bounds"]
    center_lat = (bounds[0][0] + bounds[1][0]) / 2
    center_lng = (bounds[0][1] + bounds[1][1]) / 2

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='CartoDB dark_matter',
        attr='© CARTO',
    )

    # ── Layer rendering ──
    if layer_type == "smooth_heatmap":
        # Smooth GPU-rendered heatmap from raw sample data
        heat_data = []
        metal_vals = df[metal_id].dropna()
        if len(metal_vals) > 0:
            v_min = metal_vals.min()
            v_max = metal_vals.max()
            v_range = v_max - v_min if v_max > v_min else 1
            for _, row in df.iterrows():
                val = row.get(metal_id, np.nan)
                if pd.notna(val) and pd.notna(row['lat']) and pd.notna(row['lng']):
                    weight = (val - v_min) / v_range
                    heat_data.append([row['lat'], row['lng'], weight])

            folium.plugins.HeatMap(
                heat_data,
                name=f"{METALS_INFO[metal_id]['name']} — Smooth Heatmap",
                min_opacity=0.3,
                max_val=1.0,
                radius=25,
                blur=20,
                gradient={
                    '0.0': '#1a9850',
                    '0.25': '#91cf60',
                    '0.5': '#fee08b',
                    '0.75': '#fc8d59',
                    '1.0': '#d73027',
                },
            ).add_to(m)

    elif layer_type == "health_risk":
        # Hazard Quotient: HQ = Concentration / Regulatory Limit
        heat_data = []
        if cumulative_risk:
            # Cumulative risk: sum HQ across all metals per sample
            all_metals = list(METALS_INFO.keys())
            hq_vals = []
            for _, row in df.iterrows():
                if pd.isna(row['lat']) or pd.isna(row['lng']):
                    continue
                total_hq = 0
                count = 0
                for m_id in all_metals:
                    val = row.get(m_id, np.nan)
                    if pd.notna(val):
                        total_hq += val / METALS_INFO[m_id]['limit']
                        count += 1
                if count > 0:
                    hq_vals.append(total_hq)
                    heat_data.append([row['lat'], row['lng'], total_hq])
            max_hq = max(hq_vals) if hq_vals else 1
        else:
            # Single metal HQ
            limit = METALS_INFO[metal_id]['limit']
            hq_vals = []
            for _, row in df.iterrows():
                val = row.get(metal_id, np.nan)
                if pd.notna(val) and pd.notna(row['lat']) and pd.notna(row['lng']):
                    hq = val / limit
                    hq_vals.append(hq)
                    heat_data.append([row['lat'], row['lng'], hq])
            max_hq = max(hq_vals) if hq_vals else 1

        if heat_data:
            folium.plugins.HeatMap(
                heat_data,
                name="Health Risk (Hazard Quotient)",
                min_opacity=0.35,
                max_val=max_hq,
                radius=28,
                blur=22,
                gradient={
                    '0.0': '#1a9850',
                    '0.15': '#66bd63',
                    '0.3': '#fee08b',
                    '0.5': '#fdae61',
                    '0.7': '#f46d43',
                    '0.85': '#d73027',
                    '1.0': '#67001f',
                },
            ).add_to(m)

    else:
        # Raster overlay (kriging, probability, uncertainty, realization, hotspot)
        asset_path = get_asset_path(metal_id, layer_type)
        if asset_path and os.path.exists(asset_path):
            folium.raster_layers.ImageOverlay(
                image=asset_path,
                bounds=bounds,
                opacity=opacity,
                name=f"{METALS_INFO[metal_id]['name']} — {LAYER_TYPES.get(layer_type, layer_type)}",
                interactive=True,
            ).add_to(m)

    # Sample points
    if show_points:
        fg = folium.FeatureGroup(name="Sample Points")
        metal_name = METALS_INFO[metal_id]["name"]
        limit = METALS_INFO[metal_id]["limit"]

        for _, row in df.iterrows():
            val = row.get(metal_id, np.nan)
            if pd.isna(row['lat']) or pd.isna(row['lng']):
                continue

            if pd.notna(val):
                ratio = val / limit
                if ratio > 1.5:
                    pt_color = '#ef4444'
                elif ratio > 1.0:
                    pt_color = '#f97316'
                elif ratio > 0.75:
                    pt_color = '#eab308'
                else:
                    pt_color = '#22c55e'
                popup_text = f"""
                <div style='font-family:Inter,sans-serif; font-size:12px; min-width:160px;'>
                    <b>{row.get('SampleID','—')}</b><br>
                    <b>{metal_name}:</b> {val:.1f} mg/kg<br>
                    <b>Limit:</b> {limit} mg/kg<br>
                    <b>Ratio:</b> {ratio:.2f}x
                </div>
                """
            else:
                pt_color = '#6b7280'
                popup_text = f"<b>{row.get('SampleID','—')}</b><br>No {metal_name} data"

            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=4,
                color=pt_color,
                fill=True,
                fill_color=pt_color,
                fill_opacity=0.8,
                weight=1,
                popup=folium.Popup(popup_text, max_width=250),
            ).add_to(fg)
        fg.add_to(m)

    m.fit_bounds(bounds)

    # District boundaries and labels
    if show_districts and districts_gdf is not None:
        # District type colors
        type_colors = {
            'belvaros': '#f59e0b',
            'lakotelep': '#3b82f6',
            'kertvaros': '#22c55e',
            'villanegyed': '#a855f7',
            'hagyomanyos_beepitesu_belso_lakoteruletek': '#06b6d4',
            'ipari_uzem_terulet': '#ef4444',
            'erdo': '#16a34a',
            'egyeb_belteruletek': '#64748b',
            'egyeb_ovezet': '#94a3b8',
        }

        fg_districts = folium.FeatureGroup(name="Districts", show=True)
        for _, row in districts_gdf.iterrows():
            name = row.get('nev', '')
            tipus = row.get('tipus', '')
            color = type_colors.get(tipus, '#94a3b8')

            # Build popup HTML with all metal stats for this district
            popup_rows = ""
            all_metals = ['As', 'Cd', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
            for m_id in all_metals:
                m_name = METALS_INFO[m_id]['name']
                m_limit = METALS_INFO[m_id]['limit']
                mean_val = row.get(f'{m_id}_MEAN', None)
                median_val = row.get(f'{m_id}_MEDIAN', None)
                max_val = row.get(f'{m_id}_MAX', None)
                pct95_val = row.get(f'{m_id}_PCT95', None)

                # Color based on mean vs threshold
                if mean_val is not None and not np.isnan(mean_val):
                    ratio = mean_val / m_limit
                    if ratio > 1.5:
                        badge_color = '#ef4444'
                        badge = '⚠️'
                    elif ratio > 1.0:
                        badge_color = '#f97316'
                        badge = '⚡'
                    elif ratio > 0.75:
                        badge_color = '#eab308'
                        badge = '●'
                    else:
                        badge_color = '#22c55e'
                        badge = '✓'
                    mean_str = f"{mean_val:.1f}"
                    median_str = f"{median_val:.1f}" if median_val is not None and not np.isnan(median_val) else "—"
                    max_str = f"{max_val:.1f}" if max_val is not None and not np.isnan(max_val) else "—"
                    pct95_str = f"{pct95_val:.1f}" if pct95_val is not None and not np.isnan(pct95_val) else "—"
                else:
                    badge_color = '#6b7280'
                    badge = '—'
                    mean_str = median_str = max_str = pct95_str = "—"

                popup_rows += f"""
                <tr style='border-bottom:1px solid #e2e8f0;'>
                    <td style='padding:3px 6px; font-weight:600;'>
                        <span style='color:{badge_color};'>{badge}</span> {m_id}
                    </td>
                    <td style='padding:3px 6px; text-align:right;'>{mean_str}</td>
                    <td style='padding:3px 6px; text-align:right;'>{median_str}</td>
                    <td style='padding:3px 6px; text-align:right; color:#ef4444; font-weight:600;'>{max_str}</td>
                    <td style='padding:3px 6px; text-align:right;'>{pct95_str}</td>
                    <td style='padding:3px 6px; text-align:right; color:#94a3b8;'>{m_limit}</td>
                </tr>"""

            popup_html = f"""
            <div style='font-family:Inter,Arial,sans-serif; min-width:360px; max-width:420px;'>
                <div style='background:linear-gradient(135deg,#1e293b,#0f172a); color:#f8fafc;
                            padding:10px 14px; border-radius:8px 8px 0 0;'>
                    <div style='font-size:14px; font-weight:700;'>{name}</div>
                    <div style='font-size:11px; color:#94a3b8; margin-top:2px;'>
                        {tipus.replace('_', ' ').title()}
                    </div>
                </div>
                <table style='width:100%; border-collapse:collapse; font-size:11px;
                              background:#f8fafc; color:#1e293b;'>
                    <thead>
                        <tr style='background:#e2e8f0;'>
                            <th style='padding:4px 6px; text-align:left;'>Metal</th>
                            <th style='padding:4px 6px; text-align:right;'>Mean</th>
                            <th style='padding:4px 6px; text-align:right;'>Median</th>
                            <th style='padding:4px 6px; text-align:right;'>Max</th>
                            <th style='padding:4px 6px; text-align:right;'>P95</th>
                            <th style='padding:4px 6px; text-align:right;'>Limit</th>
                        </tr>
                    </thead>
                    <tbody>{popup_rows}</tbody>
                </table>
                <div style='background:#f1f5f9; padding:4px 10px; border-radius:0 0 8px 8px;
                            font-size:10px; color:#64748b; text-align:center;'>
                    Values in mg/kg · ⚠️ &gt;1.5× · ⚡ &gt;1× · ✓ Safe
                </div>
            </div>
            """

            # District polygon
            geojson = folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    'fillColor': c,
                    'color': c,
                    'weight': 1.5,
                    'fillOpacity': 0.08,
                    'opacity': 0.6,
                },
                highlight_function=lambda x: {
                    'weight': 3,
                    'fillOpacity': 0.2,
                },
                tooltip=folium.Tooltip(
                    f"<b>{name}</b><br><i>{tipus.replace('_', ' ').title()}</i><br><small>Click for details</small>",
                    style='font-family:Inter,sans-serif; font-size:12px;'
                ),
                popup=folium.Popup(popup_html, max_width=450),
            )
            geojson.add_to(fg_districts)

            # District name label at centroid
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=f"""
                    <div style='font-family:Inter,sans-serif; font-size:10px;
                                color:rgba(255,255,255,0.85); font-weight:600;
                                text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
                                white-space:nowrap; pointer-events:none;
                                text-align:center; transform:translate(-50%,-50%);'>
                        {name}
                    </div>""",
                    icon_size=(0, 0),
                    icon_anchor=(0, 0),
                ),
            ).add_to(fg_districts)

        fg_districts.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ──────────────────────────── MAIN APP ────────────────────────────
def main():
    inject_css()

    # Load data
    df = load_data()
    config = load_config()
    districts_gdf = load_districts()

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-header'>
            <span style='font-family:Outfit; font-size:1.6rem; font-weight:800; 
                         background: linear-gradient(135deg, #10b981, #3b82f6);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                DEBRECEN WebGIS
            </span>
            <div style='color:#64748b; font-size:0.7rem; letter-spacing:1.5px; 
                        text-transform:uppercase; font-weight:600; margin-top:4px;'>
                Soil Heavy Metal Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Metal selector
        metal_id = st.selectbox(
            "🔬 SELECT ELEMENT",
            options=list(METALS_INFO.keys()),
            format_func=lambda x: f"{METALS_INFO[x]['name']} ({x})",
            index=5,  # Default to Pb
        )

        # Layer type
        layer_type = st.selectbox(
            "🗂️ ANALYSIS LAYER",
            options=list(LAYER_TYPES.keys()),
            format_func=lambda x: LAYER_TYPES[x],
        )

        st.divider()

        # Map options
        opacity = st.slider("🎚️ RASTER OPACITY", 0.1, 1.0, 0.65, 0.05)
        
        # Scientific Filtering
        st.markdown("🔍 **DATA FILTER**")
        min_val, max_val = float(df[metal_id].min()), float(df[metal_id].max())
        filter_range = st.slider(f"Show {METALS_INFO[metal_id]['name']} Range (mg/kg):", 
                                 min_val, max_val, (min_val, max_val))
        
        # Filter the dataframe for the map and stats
        df_filtered = df[(df[metal_id] >= filter_range[0]) & (df[metal_id] <= filter_range[1])].copy()
        
        show_points = st.checkbox("📍 Show Sample Points", value=True)
        show_districts = st.checkbox("🏘️ Show District Names", value=True)
        
        st.divider()
        st.markdown("🩺 **RISK SETTINGS**")
        cumulative_risk = st.checkbox("📈 Cumulative Risk Index", value=False, 
                                     help="Sum Hazard Quotients across all metals")

        st.divider()
        
        # Data Export
        st.markdown("📥 **EXPORT DATA**")
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered CSV",
            data=csv,
            file_name=f"Debrecen_{metal_id}_Filtered.csv",
            mime='text/csv',
        )

        st.divider()

        # Info card
        metal_info = METALS_INFO[metal_id]
        st.markdown(f"""
        <div class='glass-card'>
            <div style='font-size:0.75rem; color:#94a3b8; text-transform:uppercase; 
                        letter-spacing:1px; margin-bottom:8px;'>Current Target</div>
            <div style='font-size:1.3rem; font-weight:700; color:#f8fafc; 
                        font-family:Outfit;'>{metal_info['name']}</div>
            <div style='margin-top:8px; font-size:0.85rem; color:#cbd5e1;'>
                Regulatory Limit: <span style='color:#ef4444; font-weight:600;'>
                {metal_info['limit']} mg/kg</span>
            </div>
            <div style='margin-top:4px; font-size:0.85rem; color:#cbd5e1;'>
                Samples: <span style='color:#10b981; font-weight:600;'>
                {df[metal_id].notna().sum()}</span> / {len(df)}
            </div>
            <div style='margin-top:4px; font-size:0.85rem; color:#cbd5e1;'>
                Mean: <span style='font-weight:600; color:{metal_info["color"]};'>
                {df[metal_id].mean():.1f} mg/kg</span>
            </div>
            <div style='margin-top:12px;'>
                <span class='tag'>Objective 6 · Stakeholder WebGIS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Branded Hero
    st.markdown(f"""
    <div class='hero-banner'>
        <div class='hero-text'>
            <div style='font-size:0.8rem; color:#10b981; font-weight:700; 
                        text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;'>
                Scientific Research Platform
            </div>
            <h1>Debrecen Heavy Metal Dashboard</h1>
            <p style='color:#94a3b8; font-size:1rem; margin-top:8px; max-width:600px;'>
                Advanced geostatistical analysis and spatial visualization for soil contamination study 
                in the City of Debrecen.
            </p>
        </div>
        <div style='text-align:right;'>
            <div style='font-size:0.7rem; color:#64748b;'>Objective 6</div>
            <div style='font-family:Outfit; font-size:1.1rem; color:#f8fafc; font-weight:600;'>
                Stakeholder WebGIS
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_map, tab_profiles, tab_portfolio, tab_research, tab_data = st.tabs([
        "🌍 Interactive Map", "🧬 Multi-Metal Profiles", "🖼️ Portfolio Gallery", "📊 Scientific Research", "📋 Data Explorer"
    ])

    # ── TAB 1: FOLIUM MAP ──
    with tab_map:
        col_map, col_legend = st.columns([4, 1])

        with col_map:
            folium_map = create_folium_map(df_filtered, config, metal_id, layer_type, show_points, opacity, 
                                          show_districts, districts_gdf, cumulative_risk)
            st_folium(folium_map, width=None, height=600, returned_objects=[])

        with col_legend:
            st.markdown("""
            <div class='floating-legend'>
                <div style='font-size:0.75rem; color:#94a3b8; text-transform:uppercase; 
                            letter-spacing:1px; margin-bottom:12px; font-weight:700;'>
                    Spatial Legend
                </div>
                <div style='display:flex; flex-direction:column; gap:10px;'>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <div style='width:10px; height:10px; border-radius:50%; background:#ef4444; 
                                    box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);'></div>
                        <span style='color:#cbd5e1; font-size:0.75rem;'>&gt;1.5× Limit</span>
                    </div>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <div style='width:10px; height:10px; border-radius:1px; background:#f97316;'></div>
                        <span style='color:#cbd5e1; font-size:0.75rem;'>1.0—1.5× Limit</span>
                    </div>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <div style='width:10px; height:10px; border-radius:1px; background:#eab308;'></div>
                        <span style='color:#cbd5e1; font-size:0.75rem;'>0.75—1.0× Limit</span>
                    </div>
                    <div style='display:flex; align-items:center; gap:10px;'>
                        <div style='width:10px; height:10px; border-radius:50%; background:#22c55e;'></div>
                        <span style='color:#cbd5e1; font-size:0.75rem;'>&lt;0.75× Limit</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Scientific Metrics Ribbon (Below Map)
        vals = df[metal_id].dropna()
        exceeding = (vals > metal_info['limit']).sum()
        pct = exceeding / len(vals) * 100 if len(vals) > 0 else 0
        
        st.markdown(f"""
        <div class='stats-ribbon'>
            <div class='stat-item'>
                <div class='stat-label'>Contamination Rate</div>
                <div class='stat-value' style='color:#ef4444;'>{pct:.1f}%</div>
                <div style='font-size:0.7rem; color:#64748b;'>{exceeding} Samples</div>
            </div>
            <div class='stat-item'>
                <div class='stat-label'>Mean Concentration</div>
                <div class='stat-value'>{vals.mean():.1f}</div>
                <div style='font-size:0.7rem; color:#64748b;'>mg/kg</div>
            </div>
            <div class='stat-item'>
                <div class='stat-label'>Peak (Max)</div>
                <div class='stat-value'>{vals.max():.1f}</div>
                <div style='font-size:0.7rem; color:#64748b;'>mg/kg</div>
            </div>
            <div class='stat-item'>
                <div class='stat-label'>Regulatory Limit</div>
                <div class='stat-value' style='color:#94a3b8;'>{metal_info['limit']}</div>
                <div style='font-size:0.7rem; color:#64748b;'>mg/kg</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: MULTI-METAL PROFILES ──
    with tab_profiles:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='margin-top:0; font-size:1rem;'>🧬 Multi-Metal Signature Profiles</h3>
            <p style='color:#94a3b8; font-size:0.8rem; margin-bottom:0;'>
                Analyzing the chemical fingerprint of soil samples. Each line represents a unique sample's concentration 
                profile across the study elements.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Parallel Coordinates Plot (Instead of Triangles only)
        st.plotly_chart(create_parallel_coordinates(df), use_container_width=True)

        st.divider()

        col_heat, col_indiv = st.columns([1.2, 1])

        with col_heat:
            # Full population patterns
            st.plotly_chart(create_population_heatmap(df), use_container_width=True)

        with col_indiv:
            # Per-sample details
            sample_ids = df_filtered['SampleID'].dropna().tolist()
            if not sample_ids:
                st.warning("No samples match the current filter.")
            else:
                selected = st.selectbox("Detailed risk profile for sample:", sample_ids[:50])
                if selected:
                    idx = df[df['SampleID'] == selected].index[0]
                    fig_sample = create_sample_risk_bars(df, idx)
                    st.plotly_chart(fig_sample, use_container_width=True)


        # Bar comparison
        st.markdown("### Mean Concentrations vs Regulatory Limits")
        fig_bar = create_bar_comparison(df)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 3: PORTFOLIO OVERVIEW ──
    with tab_portfolio:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='margin-top:0; font-size:1.2rem;'>🖼️ Strategic Map Gallery</h3>
            <p style='color:#94a3b8; font-size:0.9rem; margin-bottom:0;'>
                Explore the complete collection of high-resolution geostatistical maps. 
                Switch to **Comparison Mode** to see patterns across different elements.
            </p>
        </div>
        """, unsafe_allow_html=True)

        portfolio_mode = st.radio(
            "Gallery View Mode:",
            ["Single Element Deep-Dive", "Cross-Element Comparison (Multi-Map)"],
            horizontal=True
        )

        # Only show static layers that have image assets
        STATIC_LAYERS = {k: v for k, v in LAYER_TYPES.items() if k not in ['smooth_heatmap', 'health_risk']}

        if portfolio_mode == "Single Element Deep-Dive":
            col_port1, col_port2 = st.columns([1, 1])
            
            with col_port1:
                port_m1 = st.selectbox("Element 1:", list(METALS_INFO.keys()), format_func=lambda x: METALS_INFO[x]['name'], index=5, key="p1_m")
                port_l1 = st.selectbox("Analysis Layer 1:", list(STATIC_LAYERS.keys()), format_func=lambda x: STATIC_LAYERS[x], index=0, key="p1_l")
                asset1 = get_asset_path(port_m1, port_l1)
                if asset1 and os.path.exists(asset1):
                    st.markdown(f"""
                    <div class='portfolio-card'>
                        <div class='portfolio-meta'>
                            <span class='portfolio-title'>{METALS_INFO[port_m1]['name']} — {STATIC_LAYERS[port_l1]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(asset1, use_container_width=True)
                else:
                    st.warning(f"⚠️ **High-Resolution Map Not Found**\nThe static '{STATIC_LAYERS[port_l1]}' layer for {METALS_INFO[port_m1]['name']} is currently being processed or missing.")

            with col_port2:
                port_m2 = st.selectbox("Element 2 (Compare):", list(METALS_INFO.keys()), format_func=lambda x: METALS_INFO[x]['name'], index=6, key="p2_m")
                port_l2 = st.selectbox("Analysis Layer 2 (Compare):", list(STATIC_LAYERS.keys()), format_func=lambda x: STATIC_LAYERS[x], index=1, key="p2_l")
                asset2 = get_asset_path(port_m2, port_l2)
                if asset2 and os.path.exists(asset2):
                    st.markdown(f"""
                    <div class='portfolio-card'>
                        <div class='portfolio-meta'>
                            <span class='portfolio-title'>{METALS_INFO[port_m2]['name']} — {STATIC_LAYERS[port_l2]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(asset2, use_container_width=True)
                else:
                    st.warning(f"⚠️ **High-Resolution Map Not Found**\nThe static '{STATIC_LAYERS[port_l2]}' layer for {METALS_INFO[port_m2]['name']} is currently being processed or missing.")

        else:
            # Cross-Element Grid
            st.markdown("### 🧬 Cross-Element Patterns")
            selected_layer = st.selectbox(
                "Filter element collection by analysis method:",
                list(STATIC_LAYERS.keys()),
                format_func=lambda x: STATIC_LAYERS[x],
                index=0
            )
            
            p_cols = st.columns(4)
            for i, m_id in enumerate(METALS_INFO.keys()):
                p_asset = get_asset_path(m_id, selected_layer)
                with p_cols[i % 4]:
                    if p_asset and os.path.exists(p_asset):
                        st.markdown(f"""
                        <div class='portfolio-card'>
                            <div class='portfolio-meta'>
                                <span class='portfolio-title'>{METALS_INFO[m_id]['name']}</span>
                                <span class='portfolio-subtitle'>{selected_layer.title()}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(p_asset, use_container_width=True)
                    else:
                        st.markdown(f"**{METALS_INFO[m_id]['name']}**")
                        st.caption("Layer pending computation")


    # ── TAB 4: SCIENTIFIC RESEARCH ──
    with tab_research:
        st.markdown("""
        <div class='glass-card' style='background:rgba(16,185,129,0.05); border-color:rgba(16,185,129,0.15);'>
            <h3 style='margin-top:0; font-size:1.2rem; color:#10b981;'>🧪 Thesis Analysis Suite</h3>
            <p style='color:#94a3b8; font-size:0.9rem; margin-bottom:0;'>
                Advanced statistical tools for investigating multi-element correlations and land-use impacts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.plotly_chart(create_correlation_matrix(df_filtered), use_container_width=True)
        
        with col_res2:
            st.plotly_chart(create_landuse_boxplots(df_filtered, districts_gdf), use_container_width=True)
            
        st.divider()
        st.markdown("### 📊 Distribution Analysis")
        m_dist = st.selectbox("Analyze Distribution for:", list(METALS_INFO.keys()), 
                              format_func=lambda x: METALS_INFO[x]['name'], key="dist_m")
        
        fig_hist = go.Figure(data=[go.Histogram(x=df_filtered[m_dist], nbinsx=30, 
                                               marker_color=METALS_INFO[m_dist]['color'])])
        fig_hist.update_layout(
            title=f"Histogram of {METALS_INFO[m_dist]['name']} (Filtered)",
            xaxis_title=METALS_INFO[m_dist]['unit'],
            yaxis_title="Count",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.3)',
            font=dict(color='#f8fafc'),
            height=320
        )
        st.plotly_chart(fig_hist, use_container_width=True)


    # ── TAB 5: DATA EXPLORER ──
    with tab_data:
        st.markdown("""
        <div class='glass-card'>
            <h3 style='margin-top:0; font-size:1rem;'>📋 Raw Data Explorer</h3>
            <p style='color:#94a3b8; font-size:0.8rem; margin-bottom:0;'>
                View and filter the XRF sampling data with spatial coordinates.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Summary statistics - ALWAYS on full dataset for scientific context
        st.markdown("### Descriptive Statistics (Full Dataset, mg/kg)")
        stats_df = df[list(METALS_INFO.keys())].describe().T
        stats_df['threshold'] = [METALS_INFO[m]['limit'] for m in stats_df.index]
        stats_df['exceed_%'] = [
            (df[m] > METALS_INFO[m]['limit']).sum() / df[m].notna().sum() * 100
            if df[m].notna().sum() > 0 else 0
            for m in stats_df.index
        ]
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

        # Raw data - use filtered
        st.markdown(f"### Sample Data (Filtered: {len(df_filtered)} samples)")
        display_cols = ['SampleID', 'lat', 'lng'] + list(METALS_INFO.keys())
        available_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df_filtered[available_cols], use_container_width=True, height=400)


if __name__ == "__main__":
    main()
