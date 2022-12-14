import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, num_of_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 12, 3, 1, 1)
        self.conv3 = nn.Conv2d(12, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_of_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out
