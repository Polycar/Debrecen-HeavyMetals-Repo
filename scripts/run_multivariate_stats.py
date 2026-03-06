import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configuration
CSV_FILE = os.path.join(PROJECT_ROOT, 'data', 'XRF_commonSpatial_Median.csv')
OUTPUT_DIR = r"d:\Debrecen-város(1)\StatisticalAnalysis"
METALS = ['Medián_As', 'Medián_Pb', 'Medián_Zn', 'Medián_Cu', 'Medián_Ni', 'Medián_Cr', 'Medián_Cd', 'Medián_Co', 'Medián_Fe', 'Medián_K', 'Medián_Mn', 'Medián_Ti', 'Medián_V']

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_multivariate_analysis():
    print("Loading data...")
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    data = df[METALS].dropna()
    
    # 1. Correlation Analysis
    print("Performing Correlation Analysis...")
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Inter-element Correlation Matrix (Spearman/Pearson)", fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "Correlation_Matrix.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. PCA
    print("Performing Principal Component Analysis (PCA)...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Variance Explained
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.5, align='center', label='Individual Variance')
    plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Cumulative Variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title('PCA: Explained Variance')
    plt.legend(loc='best')
    plt.savefig(os.path.join(OUTPUT_DIR, "PCA_Variance_ScreePlot.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. PCA Loadings (Component Matrix)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(METALS))], index=METALS)
    loadings.to_csv(os.path.join(OUTPUT_DIR, "PCA_Loadings.csv"))
    
    # Biplot (PC1 vs PC2)
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_result[:,0], pca_result[:,1], alpha=0.3, c='gray', s=10)
    
    for i, metal in enumerate(METALS):
        plt.arrow(0, 0, pca.components_[0, i]*5, pca.components_[1, i]*5, color='red', alpha=0.8, head_width=0.1)
        plt.text(pca.components_[0, i]*5.5, pca.components_[1, i]*5.5, metal.replace('Medián_', ''), color='darkred', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.xlabel(f"PC1 ({exp_var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({exp_var[1]*100:.1f}%)")
    plt.title("PCA Biplot: Element Associations and Sample Groupings", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "PCA_Biplot.png"), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Statistical Analysis completed. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_multivariate_analysis()
