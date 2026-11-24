import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()

X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Dataset Shape:", X.shape)
print("Reduced Dataset Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

print("First 5 Transformed Samples:")
print(np.round(X_pca[:5], 2))
