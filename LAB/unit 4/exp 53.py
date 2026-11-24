import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

sentences = [
    "i like machine learning",
    "i like deep learning",
    "i love machine learning",
    "i love deep learning",
    "machine learning is interesting",
    "deep learning is powerful"
]

words = sorted(set(" ".join(sentences).split()))
word_to_idx = {w: i + 1 for i, w in enumerate(words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

X = []
y = []

for sentence in sentences:
    tokens = sentence.split()
    for i in range(1, len(tokens)):
        X.append([word_to_idx[w] for w in tokens[:i]])
        y.append(word_to_idx[tokens[i]])

max_len = max(len(x) for x in X)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X = pad_sequences(X, maxlen=max_len)
y = np.array(y)

model = Sequential([
    Embedding(len(words) + 1, 16, input_length=max_len),
    SimpleRNN(32),
    Dense(len(words) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=100, verbose=0)

seed = "i like"
tokens = seed.split()

sequence = pad_sequences(
    [[word_to_idx[w] for w in tokens]], maxlen=max_len
)

prediction = model.predict(sequence, verbose=0)
next_word = idx_to_word[np.argmax(prediction)]

print("Input Sentence:", seed)
print("Predicted Next Word:", next_word)
