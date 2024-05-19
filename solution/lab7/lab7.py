from tensorflow import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
###################################
# Write your own code here #
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam


###################################
font = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

def display_random_images(x_data: np.array, y_data: np.array, count: int = 10) -> None:
    index = np.array(len(x_data))
    selected_ind = np.random.choice(index, count)

    selected_img = x_data[selected_ind]
    selected_labels = y_data[selected_ind]
    concat_img = np.concatenate(selected_img, axis=1)

    plt.figure(figsize=(20,10))
    plt.imshow(concat_img, cmap="gray")

    for id_label, label in enumerate(selected_labels):
        plt.text(14 + 28*id_label, 28*(5/4), label)
        plt.axis('off')
    
    plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#display_random_images(x_train, y_train)

# Data preprocessing
# Write your own code here #
# training data
y_train_enc = to_categorical(y_train, num_classes = 10)
X_train_exp= np.expand_dims(x_train, -1)
X_train_scaled = X_train_exp.astype('float32')/255.0
# testing data
y_test_enc = to_categorical(y_test, num_classes = 10)
X_test_exp= np.expand_dims(x_test, -1)
X_test_scaled = X_test_exp.astype('float32')/255.0

model = Sequential()
###################################
# Write your own code here #
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

loss = 'categorical_crossentropy'
learning_rate = 0.0025
optimizer = Adam(learning_rate = learning_rate)
metrics = ['accuracy']

###################################
# Write your own code here #
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
history = model.fit(X_train_scaled, y_train_enc, epochs = 50, batch_size = 512, validation_split = 0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

score = model.evaluate(X_test_scaled, y_test_enc, verbose=0)
print('Test loss:', score[0])
print(f'Test accuracy: {score[1]*100} %')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

###################################
model.summary()