import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

print("Original Dataset Shape:", X.shape)

print("\nPCA")
print("Reduced Shape:", X_pca.shape)
print("Explained Variance:",
      np.round(pca.explained_variance_ratio_, 3))

print("\nICA")
print("Reduced Shape:", X_ica.shape)
print("Independent Components Shape:", X_ica.shape)

print("\nPCA and ICA Comparison Completed")
