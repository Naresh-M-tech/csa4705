import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

data = np.sin(np.arange(1000) * 0.02)

X = []
y = []

for i in range(50, len(data)):
    X.append(data[i-50:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 50, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

models = {
    "RNN": SimpleRNN(32, input_shape=(50, 1)),
    "LSTM": LSTM(32, input_shape=(50, 1))
}

for name, layer in models.items():
    model = Sequential([
        layer,
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(X_train, y_train, epochs=5,
              batch_size=32, verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    print(name)
    print("Test Loss:", round(loss, 4))
    print("Test MAE:", round(mae, 4))
    print()
