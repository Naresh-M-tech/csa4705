import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Lambda

sentences = [
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [6, 7, 8, 9],
    [5, 7, 6, 8]
]

labels = np.array([1, 1, 1, 0, 0, 0])

X = np.array(sentences)

input_layer = Input(shape=(4,))
embedding = Embedding(10, 8)(input_layer)

w1 = Lambda(lambda x: x[:, 0, :])(embedding)
w2 = Lambda(lambda x: x[:, 1, :])(embedding)
w3 = Lambda(lambda x: x[:, 2, :])(embedding)
w4 = Lambda(lambda x: x[:, 3, :])(embedding)

merge = Concatenate()

left = Dense(16, activation='tanh')(merge([w1, w2]))
right = Dense(16, activation='tanh')(merge([w3, w4]))

root = Dense(16, activation='tanh')(merge([left, right]))
output = Dense(1, activation='sigmoid')(root)

model = Model(input_layer, output)

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, labels, epochs=50, verbose=0)

predictions = model.predict(X, verbose=0)

for p in predictions:
    print("Positive" if p[0] > 0.5 else "Negative")
