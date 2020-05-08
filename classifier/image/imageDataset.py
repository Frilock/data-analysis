import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from skimage import io, transform
from torch.utils.data.sampler import SubsetRandomSampler


class ImageDataset(Dataset):
    def __init__(self):
        self.root_dir = '../../images/imageClassifier'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_marks = pd.read_csv('../../csv/classifier/images/t2_x_train.csv')
        self.test_marks = pd.read_csv('../../csv/classifier/images/t2_x_test.csv')

    def __len__(self):
        return int(len(self.train_marks))

    def __getitem__(self, item):
        sample = io.imread(self.train_marks[item])
        return self.transform(sample)
