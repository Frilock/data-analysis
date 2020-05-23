import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, AveragePooling2D, Dropout, Input, Add, Activation
from keras.optimizers import SGD

image_size = 227
depth = 3
classes = 10


def res_net():
    input_image = Input(shape=(image_size, image_size, depth))

    x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
               padding="valid", activation="relu")(input_image)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)

    x_shortcut = x

    x = Conv2D(filters=384, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")(x)
    x = Conv2D(filters=384, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")(x)

    x = Dropout(0.5)(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(9216, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)

    output = Dense(classes, activation="sigmoid")(x)

    opt = SGD(lr=0.001, momentum=0.9)

    model = Model(inputs=input_image, outputs=output)
    model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    return model
