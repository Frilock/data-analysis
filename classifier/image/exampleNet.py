import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from types import SimpleNamespace

# FLAGS
args = SimpleNamespace(
    batch_size=64,
    test_batch_size=1000,
    lr=1,
    gamma=0.7,
    log_interval=10000,
    epochs=14,
    no_cuda=False,
)

use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")


class LeNet(torch.nn.Module):
    def __init__(self, h, w, c):
        torch.nn.Module.__init__(self)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )

        len_features = h // 4 * w // 4 * 64

        self.classifier = nn.Sequential(
            nn.Linear(len_features, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
            nn.Sigmoid(),
        )

    def forward(self, input_):
        features = self.feature_extractor(input_)
        features = torch.flatten(features, 1)
        class_labels = self.classifier(features)
        return class_labels


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

os.makedirs("./mnist", exist_ok=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

print(f"Number of train batches (batch_size={args.batch_size}):", len(train_loader))
print(f"Number of test batches (batch_size={args.test_batch_size}):", len(test_loader))

print(f"Number of train samples: {args.batch_size * len(train_loader)}")
print(f"Number of test samples: {args.test_batch_size * len(test_loader)}")


def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


model = LeNet(28, 28, 1).to(device=device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, loss_fn, epoch)
    test(model, device, test_loader, loss_fn)
    scheduler.step()
