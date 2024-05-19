import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 0])
# model = Sequential()
# model.add(Dense(2, input_dim=2, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# history = model.fit(X, y, epochs=1000, batch_size=1, verbose=1)
# loss, accuracy = model.evaluate(X, y, verbose=1)
# print('Accuracy: {:.2f}'.format(accuracy*100))

path_to_dataset = 'C:/Users/Artem/Study/VUT_FEKT/MLF/mpa-mlf-exercises/Lab_05/voting_complete.csv' # change the PATH
pd_dataset = pd.read_csv(path_to_dataset)

# define a function for train and test split
def train_test_split(pd_data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    pd_dataset = pd_data.copy()
    pd_dataset = pd_dataset[pd_dataset.columns[1:]]
    index = np.arange(len(pd_dataset))
    index = np.random.permutation(index)
    train_ammount = int(len(index)*test_ratio)
    train_ids = index[train_ammount:]
    test_ids = index[:train_ammount]

    train_dataset = pd_dataset[pd_dataset.index.isin(train_ids)].reset_index()
    test_dataset = pd_dataset[pd_dataset.index.isin(test_ids)].reset_index()
    
    train_dataset = train_dataset[train_dataset.columns[1:]]
    test_dataset = test_dataset[test_dataset.columns[1:]]

    return train_dataset[train_dataset.columns[1:]], train_dataset[train_dataset.columns[0]], test_dataset[test_dataset.columns[1:]], test_dataset[test_dataset.columns[0]]

x_train, y_train, x_test, y_test = train_test_split(pd_dataset)
print(x_train)

pd_dataset.replace('?', np.nan, inplace=True)
pd_dataset.replace('y', 1, inplace=True)
pd_dataset.replace('n', 0, inplace=True)
pd_dataset.replace('republican', 1, inplace=True)
pd_dataset.replace('democrat', 0, inplace=True)
pd_dataset = pd_dataset.drop(columns=['export-administration-act-south-africa'])

x_train, y_train, x_test, y_test = train_test_split(pd_dataset)
print(pd_dataset)

columns = x_train.columns
for column in columns:
    pd_dataset[column] = pd_dataset[column].fillna(x_train[column].value_counts().argmax())
    pd_dataset[column] = pd_dataset[column].astype('int')

model = Sequential()
model.add(Dense(4, input_dim=len(columns), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=1000, batch_size=32, verbose=1)
loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print('Accuracy: {:.2f}'.format(accuracy*100))

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Accuracy: {:.2f}'.format(accuracy*100))
print(history.history.keys())

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('n epochs')
plt.ylabel('loss & validation loss')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('n epochs')
plt.ylabel('accuracy & validation accuracy')

plt.show()