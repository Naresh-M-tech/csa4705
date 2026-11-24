import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_iris()

X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))

for i, name in enumerate(data.target_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=name
    )

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Visualization of Iris Dataset")
plt.legend()
plt.grid()
plt.show()

print("Original Features:", X.shape[1])
print("Reduced Features:", X_pca.shape[1])
print("Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")
