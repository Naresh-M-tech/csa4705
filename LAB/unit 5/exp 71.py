from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()

X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

variance = pca.explained_variance_ratio_
cumulative_variance = variance.cumsum()

components = next(
    i + 1 for i, value in enumerate(cumulative_variance)
    if value >= 0.95
)

print("Original Features:", X.shape[1])
print("Required Components:", components)
print("Cumulative Explained Variance:",
      round(cumulative_variance[components - 1] * 100, 2), "%")
print("Reduced Dataset Shape:",
      (X.shape[0], components))
