"""
titanic_kmeans_pca.py
From-scratch PCA + K-Means clustering on Titanic dataset using two features.
Default features: 'Age' and 'Fare' (changeable).
Save results to titanic_clustered.csv and show clustering plots.

Requirements:
- Python 3.8+
- pandas, numpy, matplotlib

Usage:
- Place titanic.csv or titanic.xlsx in the same folder (CSV preferred).
- Run: python titanic_kmeans_pca.py
"""

import os
import sys
import random
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config / Parameters (change if you want)
# -----------------------------
FEATURES = ['Age', 'Fare']   # two features to use for clustering
K = 3                       # number of clusters for K-means
MAX_ITERS = 300
TOL = 1e-6                  # tolerance for centroid movement convergence
RANDOM_SEED = 42

# -----------------------------
# Utility functions
# -----------------------------
def load_titanic_data() -> pd.DataFrame:
    """Load titanic.csv or titanic.xlsx from current dir."""
    csv_path = 'titanic.csv'
    xlsx_path = 'titanic.xlsx'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path} (rows={len(df)})")
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
        print(f"Loaded data from {xlsx_path} (rows={len(df)})")
    else:
        raise FileNotFoundError("No 'titanic.csv' or 'titanic.xlsx' found in current directory. Please place your dataset file here.")
    return df

def preprocess(df: pd.DataFrame, features: list) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Select features, handle missing values (median imputation), and standardize.
    Returns standardized numpy array and the processed DataFrame slice (before standardize).
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Feature columns not found in dataset: {missing}")

    X_df = df[features].copy()

    # Impute missing values with median
    for col in features:
        med = X_df[col].median()
        X_df[col].fillna(med, inplace=True)
        print(f"Imputed missing values in '{col}' with median = {med}")

    # Convert to numeric if needed (coerce errors)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    # Re-impute any remaining NaNs with median
    for col in features:
        if X_df[col].isna().any():
            med = X_df[col].median()
            X_df[col].fillna(med, inplace=True)
            print(f"Post-conversion imputed remaining NaNs in '{col}' with median = {med}")

    # Standardize (zero mean, unit variance)
    means = X_df.mean()
    stds = X_df.std(ddof=0)  # population std
    X_std = (X_df - means) / stds.replace(0, 1)  # avoid div-by-zero

    print("Standardization: means =", means.to_dict())
    print("Standardization: stds  =", stds.to_dict())

    return X_std.values, X_df

# -----------------------------
# PCA from scratch
# -----------------------------
def pca_from_scratch(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA:
    - center data (mean zero)
    - compute covariance matrix (features x features)
    - eigen-decomposition
    - sort eigenpairs by eigenvalue descending
    - return:
        - projected data (n_samples, n_components)
        - principal components matrix (n_components, n_features)
        - explained variance array (length n_components)
    """
    # Ensure X is shape (n_samples, n_features)
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape

    # Centering
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    # Covariance matrix (features x features), use unbiased estimator: divide by (n_samples - 1)
    cov_matrix = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)  # eigh for symmetric matrices
    # eig_vals ascending -> sort descending
    idx_sorted = np.argsort(eig_vals)[::-1]
    eig_vals_sorted = eig_vals[idx_sorted]
    eig_vecs_sorted = eig_vecs[:, idx_sorted]

    # Select top components
    components = eig_vecs_sorted[:, :n_components].T  # shape (n_components, n_features)
    explained_variance = eig_vals_sorted[:n_components]
    # Project
    X_pca = np.dot(X_centered, components.T)  # (n_samples, n_components)

    print("PCA: eigenvalues (top) =", explained_variance)
    total_variance = eig_vals_sorted.sum()
    explained_ratio = explained_variance / total_variance
    print("PCA: explained variance ratio (top) =", explained_ratio)

    return X_pca, components, explained_variance

# -----------------------------
# K-Means from scratch
# -----------------------------
def initialize_centroids(X: np.ndarray, k: int, random_state: int = None) -> np.ndarray:
    """Initialize centroids by randomly choosing k distinct points from X."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices, :].astype(float)
    return centroids

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each sample to the nearest centroid (Euclidean)."""
    # Compute squared distances (n_samples, k)
    dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
    labels = np.argmin(dists, axis=1)
    return labels

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Update centroids as the mean of assigned points. If a centroid has no points, reinitialize randomly."""
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    for i in range(k):
        points = X[labels == i]
        if len(points) == 0:
            # Reinitialize centroid randomly (choose random sample)
            centroids[i] = X[np.random.randint(0, X.shape[0])]
            print(f"Centroid {i} had no points; reinitialized.")
        else:
            centroids[i] = points.mean(axis=0)
    return centroids

def kmeans_from_scratch(X: np.ndarray, k: int, max_iters: int = 300, tol: float = 1e-6, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-means clustering:
    Returns labels (n_samples,), centroids (k, n_features), inertia (sum squared distance within clusters)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    centroids = initialize_centroids(X, k, random_state=random_state)
    for it in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        movement = np.sqrt(np.sum((centroids - new_centroids)**2, axis=1)).max()
        centroids = new_centroids
        # print progress occasionally
        if it % 10 == 0 or it == max_iters - 1:
            print(f"KMeans iter {it}: centroid max movement = {movement:.6f}")
        if movement <= tol:
            print(f"KMeans converged at iteration {it} with movement {movement:.6f}")
            break

    # compute inertia
    final_dists = np.sum((X - centroids[labels])**2, axis=1)
    inertia = final_dists.sum()
    print(f"KMeans final inertia (sum squared distances) = {inertia:.4f}")

    return labels, centroids, inertia

# -----------------------------
# Visualization
# -----------------------------
def plot_clusters_2d(X_orig: np.ndarray, X_std: np.ndarray, labels: np.ndarray, centroids: np.ndarray, features: list):
    """
    X_orig: original (unstandardized) values shape (n_samples, 2)
    X_std: standardized values shape (n_samples, 2)
    labels: cluster labels
    centroids: centroids in standardized space (k, 2)
    """
    plt.figure(figsize=(10, 5))

    # Left: original feature space scatter
    plt.subplot(1, 2, 1)
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(X_orig[mask, 0], X_orig[mask, 1], label=f'Cluster {c}', alpha=0.6)
    # Plot centroids converted back to original scale for display (centroid_orig = centroid_std * std + mean)
    # But we don't have std/mean passed here; centroid in standardized space -> convert by ref below when calling
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Clusters in original feature space')
    plt.legend()

    # Right: standardized (PCA) space scatter
    plt.subplot(1, 2, 2)
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(X_std[mask, 0], X_std[mask, 1], label=f'Cluster {c}', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='k', label='Centroids')
    plt.xlabel(f'{features[0]} (standardized)')
    plt.ylabel(f'{features[1]} (standardized)')
    plt.title('Clusters (standardized feature space)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_pca_components(X_pca: np.ndarray, labels: np.ndarray, explained_variance: np.ndarray):
    """Plot data projected onto PCA components (first two components)."""
    plt.figure(figsize=(6,5))
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {c}', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    evr = explained_variance / explained_variance.sum()
    title = f'PCA projection (explained var top2: {evr[0]:.2f}, {evr[1]:.2f})' if len(evr)>1 else 'PCA projection'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main flow
# -----------------------------
def main():
    # Load data
    try:
        df = load_titanic_data()
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # Preprocess
    try:
        X_std, X_orig_df = preprocess(df, FEATURES)
    except Exception as e:
        print("Preprocessing error:", e)
        sys.exit(1)

    # If user chose two features, shape is (n_samples, 2). We'll run PCA (n_components=2) and K-means on standardized features.
    n_components = min(2, X_std.shape[1])
    X_pca, components, explained_variance = pca_from_scratch(X_std, n_components=n_components)

    # Run kmeans on standardized features (not the original)
    labels, centroids, inertia = kmeans_from_scratch(X_std, k=K, max_iters=MAX_ITERS, tol=TOL, random_state=RANDOM_SEED)

    # Save results to CSV
    out_df = df.copy()
    out_df['_cluster_label'] = labels
    # Add PCA columns
    for i in range(n_components):
        out_df[f'PC{i+1}'] = X_pca[:, i]
    out_csv = 'titanic_clustered.csv'
    out_df.to_csv(out_csv, index=False)
    print(f"Saved clustered results to {out_csv}")

    # Visualize clusters: need both original (unstandardized) and standardized arrays
    X_orig_vals = X_orig_df.values  # original numeric (but still imputed)
    plot_clusters_2d(X_orig_vals, X_std, labels, centroids, FEATURES)

    # Plot PCA projection colored by cluster
    if n_components >= 2:
        plot_pca_components(X_pca, labels, explained_variance)

    # Print brief summary
    print("\nSummary:")
    print(f"- Features used: {FEATURES}")
    print(f"- K (clusters): {K}")
    print(f"- Inertia: {inertia:.4f}")
    tv = explained_variance.sum()
    print(f"- PCA top {n_components} explained variance (sum) = {tv:.4f}")

if __name__ == '__main__':
    main()
