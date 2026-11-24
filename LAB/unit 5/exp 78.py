import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

mean = [50, 60, 70]
covariance = [
    [10, 2, 3],
    [2, 15, 4],
    [3, 4, 20]
]

data = np.random.multivariate_normal(mean, covariance, 500)

estimated_mean = np.mean(data, axis=0)
estimated_covariance = np.cov(data, rowvar=False)

print("Generated Data Shape:", data.shape)
print("\nEstimated Mean:")
print(np.round(estimated_mean, 2))

print("\nEstimated Covariance Matrix:")
print(np.round(estimated_covariance, 2))

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multivariate Gaussian Distribution")
plt.grid()
plt.show()
