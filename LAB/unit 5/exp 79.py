import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

group1 = np.random.normal([30, 2000], [5, 300], (100, 2))
group2 = np.random.normal([60, 5000], [8, 500], (100, 2))
group3 = np.random.normal([90, 8000], [5, 400], (100, 2))

X = np.vstack([group1, group2, group3])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)
probabilities = gmm.predict_proba(X_scaled)

print("Total Customers:", len(X))
print("Number of Clusters:", len(np.unique(labels)))
print("\nCluster Distribution:")
print(np.bincount(labels))

print("\nMembership Probabilities (First 5 Customers):")
print(np.round(probabilities[:5], 3))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Customer Age")
plt.ylabel("Purchase Amount")
plt.title("GMM Customer Segmentation")
plt.show()
