import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

source = ["hello", "how are you", "good morning", "thank you"]
target = ["hola", "como estas", "buenos dias", "gracias"]

source_words = sorted(set(" ".join(source).split()))
target_words = sorted(set(" ".join(target).split()))

src = {w: i + 1 for i, w in enumerate(source_words)}
tgt = {w: i + 1 for i, w in enumerate(target_words)}

encoder_data = np.zeros((4, 3), dtype='int32')
decoder_data = np.zeros((4, 3), dtype='int32')
target_data = np.zeros((4, 3, len(tgt) + 1))

for i, (s, t) in enumerate(zip(source, target)):
    sw = s.split()
    tw = t.split()

    encoder_data[i, :len(sw)] = [src[w] for w in sw]
    decoder_data[i, 0] = 1
    decoder_data[i, 1:len(tw)+1] = [tgt[w] for w in tw]

    for j, word in enumerate(tw):
        target_data[i, j, tgt[word]] = 1

encoder_input = Input(shape=(3,))
encoder_embedding = Embedding(len(src) + 1, 16)(encoder_input)
_, state_h, state_c = LSTM(32, return_state=True)(encoder_embedding)

decoder_input = Input(shape=(3,))
decoder_embedding = Embedding(len(tgt) + 1, 16)(decoder_input)

decoder_lstm = LSTM(32, return_sequences=True)
decoder_output = decoder_lstm(
    decoder_embedding, initial_state=[state_h, state_c]
)

output = Dense(len(tgt) + 1, activation='softmax')(decoder_output)

model = Model([encoder_input, decoder_input], output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([encoder_data, decoder_data], target_data,
          epochs=200, verbose=0)

print("Translation Model Training Completed")
print("Source Language: English")
print("Target Language: Spanish")
print("Training Accuracy:", round(
    model.evaluate([encoder_data, decoder_data],
                   target_data, verbose=0)[1] * 100, 2), "%")
