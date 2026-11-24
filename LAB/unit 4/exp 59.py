import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import SimpleRNN, LSTM

X = tf.random.normal((1, 100, 1))

rnn = SimpleRNN(16)
lstm = LSTM(16)

with tf.GradientTape() as tape:
    tape.watch(X)
    output = rnn(X)
    loss = tf.reduce_sum(output)

grad_rnn = tape.gradient(loss, X)

with tf.GradientTape() as tape:
    tape.watch(X)
    output = lstm(X)
    loss = tf.reduce_sum(output)

grad_lstm = tape.gradient(loss, X)

print("RNN Gradient at First Time Step:",
      tf.norm(grad_rnn[:, 0, :]).numpy())
print("RNN Gradient at Last Time Step:",
      tf.norm(grad_rnn[:, -1, :]).numpy())

print("\nLSTM Gradient at First Time Step:",
      tf.norm(grad_lstm[:, 0, :]).numpy())
print("LSTM Gradient at Last Time Step:",
      tf.norm(grad_lstm[:, -1, :]).numpy())
