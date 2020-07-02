import cv2
import numpy as np

image_directory = '../../../resources/images/imageClassifier/'


def image_loader(file_names, image_size):  # преобразуем входной массив данных в числа
    data = []
    for file_name in file_names:
        image = cv2.imread(image_directory + file_name[0])  # нулевой индекс, потому что это массив из 1 элемента
        image = cv2.resize(image, (image_size, image_size))
        data.append(image)
    return np.array(data, dtype="float")
