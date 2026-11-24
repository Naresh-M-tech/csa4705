import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

np.random.seed(42)

time = np.linspace(0, 10, 2000)

voice1 = np.sin(2 * np.pi * 5 * time)
voice2 = np.sign(np.sin(2 * np.pi * 3 * time))

sources = np.c_[voice1, voice2]

mixing_matrix = np.array([[1, 0.5], [0.5, 1]])
mixed = sources @ mixing_matrix.T

ica = FastICA(n_components=2, random_state=42)
separated = ica.fit_transform(mixed)

print("Original Signals Shape:", sources.shape)
print("Mixed Signals Shape:", mixed.shape)
print("Separated Signals Shape:", separated.shape)

plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.plot(time, sources)
plt.title("Original Voice Signals")

plt.subplot(3, 1, 2)
plt.plot(time, mixed)
plt.title("Mixed Microphone Signals")

plt.subplot(3, 1, 3)
plt.plot(time, separated)
plt.title("ICA Separated Signals")

plt.tight_layout()
plt.show()
