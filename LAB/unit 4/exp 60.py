import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

data = np.sin(np.arange(500) * 0.1)

for seq_length in [5, 10, 20]:
    X = []
    y = []

    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])

    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential([
        SimpleRNN(16, input_shape=(seq_length, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    print("Sequence Length:", seq_length)
    print("Test Loss:", round(loss, 4))
    print("Test MAE:", round(mae, 4))
    print()
