import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam


# def generate_time_series(batch_size, n_steps):
#   freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
#   time = np.linspace(0,1,n_steps)

#   series = 0.5 * np.sin((time - offsets1) * (freq1*10 + 10))
#   series += 0.2 * np.sin((time - offsets2) * (freq2*20 + 20))
#   series += 0.1 * (np.random.rand(batch_size, n_steps)- 0.5)
#   return series[..., np.newaxis].astype(np.float32)

# def plot_function(history):
#   plt.figure()

#   plt.plot(history.history['loss'], label='loss')
#   plt.plot(history.history['val_loss'], label='validation_loss')
#   plt.legend()
#   plt.grid()
#   plt.xlim([0,no_training_epochs-1])
#   plt.xlabel('epochs')


# data_size = 10000
# n_steps = 50
# np.random.seed(0)

# # Generating dataset
# series = generate_time_series(data_size, n_steps + 1)
# # Train/Valid/Test split
# X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
# X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
# X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
# no_training_epochs = 5
# # Data examination
# # plt.figure()
# # plt.plot(X_train[0])
# # plt.plot(n_steps+1, y_train[0], 'rx')

# # Simple Linear model

# # Model definition
# model_linear = Sequential()
# model_linear.add(Flatten(input_shape=(50,1)))
# model_linear.add(Dense(1, activation = None))
# # Model building
# learning_rate = 0.001
# optimizer = Adam(learning_rate)
# model_linear.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
# model_linear.summary()
# # Model training
# history_linear = model_linear.fit(X_train, y_train, epochs=no_training_epochs, validation_data=[X_valid, y_valid])

# plot_function(history_linear)
# score = model_linear.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print(f'Test MSE: ', score[1])

# # Simple RNN Model

# # Model definition
# model_simple_rnn = Sequential()
# model_simple_rnn.add(SimpleRNN(1, input_shape=[50, 1]))
# # Model building
# learning_rate_simple_rnn = 0.001
# optimizer_simple_rnn = Adam(learning_rate_simple_rnn)
# model_simple_rnn.compile(loss='mean_squared_error', optimizer=optimizer_simple_rnn, metrics=['mean_squared_error'])
# history_simple_rnn = model_simple_rnn.fit(X_train, y_train, epochs=no_training_epochs, validation_data=[X_valid, y_valid])
# plot_function(history_simple_rnn)

# # Deep RNN Model

# # Model definition
# model_deep_rnn = Sequential()
# model_deep_rnn.add(SimpleRNN(20, return_sequences = True, input_shape=[50, 1]))
# model_deep_rnn.add(SimpleRNN(20, return_sequences = True))
# model_deep_rnn.add(SimpleRNN(1, input_shape=[50, 1]))
# # Model building
# learning_rate_deep_rnn = 0.001
# optimizer_deep_rnn = Adam(learning_rate_deep_rnn)
# model_deep_rnn.compile(loss='mean_squared_error', optimizer=optimizer_deep_rnn, metrics=['mean_squared_error'])
# history_deep_rnn = model_deep_rnn.fit(X_train, y_train, epochs=no_training_epochs, validation_data=[X_valid, y_valid])
# score = model_deep_rnn.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print(f'Test MSE: ', score[1])
# plot_function(history_simple_rnn)


import keras
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SimpleRNN, GRU
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3:word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
  id_to_word[id_] = token
single_sequence = " ".join([id_to_word[id_] for id_ in X_train[0][:]])
max_length = 150  # Define the maximum length of a review
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post', value=0)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post', value=0)

embed_size = 256
vocabulary_size = 2000
rnn_model = keras.models.Sequential([
    keras.layers.Embedding(vocabulary_size, embed_size,
                           input_shape=[None]),
    keras.layers.SimpleRNN(128, return_sequences=True),
    keras.layers.SimpleRNN(128),
    keras.layers.Dense(1, activation='sigmoid')
    ])
lstm_model = keras.models.Sequential([
    keras.layers.Embedding(vocabulary_size, embed_size,
                           input_shape=[None]),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(128),
    keras.layers.Dense(1, activation='sigmoid')
    ])
gru_model = keras.models.Sequential([
    keras.layers.Embedding(vocabulary_size, embed_size,
                           input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation='sigmoid')
    ])

# RNN
optimizer = Adam(learning_rate = 0.001)
rnn_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = rnn_model.fit(X_train, y_train, epochs=5, validation_split=0.2)
score = rnn_model.evaluate(X_test, y_test, verbose=0)
print('Test loss RNN:', score[0]) # 0.6931447982788086
print(f'Test accuracy RNN: ', score[1]*100, "%") # 50.0 %
# LSTM
optimizer = Adam(learning_rate = 0.001)
lstm_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = lstm_model.fit(X_train, y_train, epochs=5, validation_split=0.2)
score = lstm_model.evaluate(X_test, y_test, verbose=0)
print('Test loss LSTM:', score[0]) # 0.3985774517059326
print(f'Test accuracy LSTM: ', score[1]*100, "%") # 82.58799910545349 %
# GRU
optimizer = Adam(learning_rate = 0.001)
gru_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = gru_model.fit(X_train, y_train, epochs=5, validation_split=0.2)
score = gru_model.evaluate(X_test, y_test, verbose=0)
print('Test loss GRU:', score[0]) # 0.4124431908130646
print(f'Test accuracy GRU: ', score[1]*100, "%") # 83.46400260925293 %

plt.show()