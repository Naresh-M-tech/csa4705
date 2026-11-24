import numpy as np

image = np.array([
    [1, 2, 0, 3, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [1, 2, 3, 4, 5],
    [2, 1, 0, 1, 2]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

feature_map = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        feature_map[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)

pooling = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        pooling[i, j] = np.max(feature_map[i:i+2, j:j+2])

print("Input Image:")
print(image)

print("\nConvolution Feature Map:")
print(feature_map)

print("\nMax Pooling Output:")
print(pooling)
