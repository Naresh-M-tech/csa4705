import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    [1, 2, 3], [2, 3, 4], [1, 4, 5], [3, 4, 5],
    [6, 7, 8], [7, 8, 9], [6, 8, 9], [7, 9, 10]
]

labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

X = pad_sequences(texts, maxlen=5)

model = Sequential([
    Embedding(input_dim=11, output_dim=8, input_length=5),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, labels, epochs=20, verbose=0)

predictions = model.predict(X, verbose=0)

for i in range(len(texts)):
    sentiment = "Positive" if predictions[i][0] > 0.5 else "Negative"
    print(sentiment)
