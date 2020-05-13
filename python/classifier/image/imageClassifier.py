# Необходимо обучить классификатор изображений с пересекающимися классами объектов
#
# Провести сравнительный анализ архитектуры ResNet и "классической" сверточной сети.
# Для этого реализовать и обучить пятислойную сверточную нейронную сеть  с архитектурой Alexnet или VGG.
# Затем добавить в нее skip-связи, чтобы получить архитектуру ResNet.
# Провести сравнительный анализ реализованных сетей.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from python.classifier.image.alexNet import alex_net
from python.classifier.image.imageUtils import image_loader
from sklearn import metrics

generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                               height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                               horizontal_flip=True, fill_mode='nearest')
epochs = 20
batch_size = 64
classes = 10
image_size = 227


def main():
    x_train = pd.read_csv('../../../resources/csv/classifier/images/t2_x_train.csv', delimiter=',', header=None)
    y_train = pd.read_csv('../../../resources/csv/classifier/images/t2_y_train.csv').to_numpy()
    x_test = pd.read_csv('../../../resources/csv/classifier/images/t2_x_test.csv', delimiter=',', header=None)

    x_train = image_loader(np.array(x_train), image_size)  # возможно, сохранить метки
    x_test = image_loader(np.array(x_test), image_size)

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = alex_net(epochs)

    # model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),
    #                    validation_data=(x_validate, y_validate),
    #                    epochs=epochs)

    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_validate, y_validate), epochs=epochs)

    print("f1_score:", metrics.f1_score(y_validate,
                                        np.round(model.predict(x_validate, batch_size=batch_size)).astype(int),
                                        average='macro'))

    y_test = pd.DataFrame(np.round(model.predict(x_test, batch_size=batch_size)).astype(int))
    print("y_test:", y_test)

    y_test.to_csv('../../../resources/csv/classifier/images/lab4.csv', index=False, header=False)


main()
