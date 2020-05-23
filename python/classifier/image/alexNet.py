import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam, SGD


def alex_net():
    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(227, 227, 3), kernel_size=(11, 11),
                     strides=(4, 4), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(Flatten())
    model.add(Dense(9216, activation="relu"))
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(10, activation="sigmoid"))

    # opt = Adam(lr=0.001, decay= 0.001 / epochs)
    opt = SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    return model
