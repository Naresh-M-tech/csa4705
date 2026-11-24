import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import mean_squared_error

data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_reconstructed = pca.inverse_transform(X_pca)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)
fa_reconstructed = X_fa @ fa.components_ + fa.mean_

pca_error = mean_squared_error(X_scaled, pca_reconstructed)
fa_error = mean_squared_error(X_scaled, fa_reconstructed)

print("Original Dataset Shape:", X.shape)

print("\nPCA")
print("Reduced Shape:", X_pca.shape)
print("Reconstruction Error:", round(pca_error, 4))

print("\nFactor Analysis")
print("Reduced Shape:", X_fa.shape)
print("Reconstruction Error:", round(fa_error, 4))

print("\nComparison Completed")
