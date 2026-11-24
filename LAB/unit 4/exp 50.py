import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

text = "hello world hello world hello world"

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 5
X = []
y = []

for i in range(len(text) - seq_length):
    X.append([char_to_idx[c] for c in text[i:i+seq_length]])
    y.append(char_to_idx[text[i+seq_length]])

X = np.array(X)
y = np.array(y)

model = Sequential([
    Embedding(len(chars), 16),
    SimpleRNN(32),
    Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=100, verbose=0)

seed = "hello"
generated = seed

for _ in range(10):
    sequence = np.array([[char_to_idx[c] for c in generated[-seq_length:]]])
    prediction = model.predict(sequence, verbose=0)
    next_char = idx_to_char[np.argmax(prediction)]
    generated += next_char

print("Generated Text:", generated)
