import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

np.random.seed(42)

time = np.linspace(0, 8, 1000)

source1 = np.sin(2 * time)
source2 = np.sign(np.sin(3 * time))

S = np.c_[source1, source2]
S = S / S.std(axis=0)

A = np.array([[1, 1], [0.5, 2]])

mixed = S @ A.T

ica = FastICA(n_components=2, random_state=42)
separated = ica.fit_transform(mixed)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time, S)
plt.title("Original Signals")

plt.subplot(3, 1, 2)
plt.plot(time, mixed)
plt.title("Mixed Signals")

plt.subplot(3, 1, 3)
plt.plot(time, separated)
plt.title("Separated Signals")

plt.tight_layout()
plt.show()

print("Original Signal Shape:", S.shape)
print("Mixed Signal Shape:", mixed.shape)
print("Separated Signal Shape:", separated.shape)
print("ICA Signal Separation Completed")
