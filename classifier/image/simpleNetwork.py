# Во второй работе необходимо обучить классификатор изображений с пересекающимися классами объектов
#
# Провести сравнительный анализ архитектуры ResNet и "классической" сверточной сети.
# Для этого реализовать и обучить пятислойную сверточную нейронную сеть  с архитектурой Alexnet или VGG.
# Затем добавить в нее skip-связи, чтобы получить архитектуру ResNet.
# Провести сравнительный анализ реализованных сетей.
#
# Условие сдачи работы: значение F1-score должно превысить порог 0.2 на тестовой выборке.
# Чтобы достичь этот порог рекомендуется в качестве класса алгоритмов взять сверточные нейронные сети (CNN).

import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from classifier.image.imageDataset import ImageDataset

EPOCHS = 2000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRANS = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.folder.Image.Image()


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

    def forward(self):
        print("")


dataset = ImageDataset()
print(dataset.__getitem__(1))
