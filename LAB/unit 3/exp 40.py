import numpy as np

image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

max_pool = np.zeros((2, 2))
avg_pool = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        block = image[i*2:i*2+2, j*2:j*2+2]
        max_pool[i, j] = np.max(block)
        avg_pool[i, j] = np.mean(block)

print("Input Image:")
print(image)

print("\nMax Pooling:")
print(max_pool)

print("\nAverage Pooling:")
print(avg_pool)
