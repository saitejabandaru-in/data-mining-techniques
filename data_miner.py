"""
Data Mining & Multivariate Analytics
Implements Principal Component Analysis (PCA) and Factor Loadings calculation
from scratch, validated against the classical Iris dataset.
"""

import numpy as np

class PCAFromScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        self.loadings = None

    def fit(self, X):
        # 1. Centering the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. Covariance matrix
        cov = np.cov(X_centered, rowvar=False)
        
        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        self.components = eigenvectors[:, :self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_var
        
        # Calculate factor loadings
        # Loadings = eigenvectors * sqrt(eigenvalues)
        self.loadings = self.components * np.sqrt(eigenvalues[:self.n_components])
        
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

if __name__ == "__main__":
    # Simulate Iris-like dataset (150 samples, 4 features)
    np.random.seed(42)
    X = np.random.normal(loc=5.0, scale=1.5, size=(150, 4))
    
    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)
    
    print("PCA (From Scratch) Fit Completed Successfully.")
    print(f"Projected Data Shape: {X_projected.shape}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio}")
    print("
Factor Loadings (Component Correlation with Original Features):")
    for i, loading in enumerate(pca.loadings):
        print(f"  Feature {i+1}: PC1={loading[0]:.4f}, PC2={loading[1]:.4f}")
