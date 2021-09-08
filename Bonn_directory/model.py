# import

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

# LinearNet

class LinearNet(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        super(LinearNet, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i]) for i in range(1, num_layers-1)])
        self.linears.append(nn.Linear(layers_size[num_layers-2], output_size))

    def forward(self, x):
        return x


class CNN(torch.nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = torch.nn.Conv1d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 3, 1)

        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(130944, 128)
        self.fc2 = torch.nn.Linear(128, 3)

        self.dropout = None
        self.linear_array = LinearNet(input_size=32640, num_layers=args.layers, layers_size=args.layer_features,
                                      output_size = 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        self.dropout1 = nn.Dropout(self.args.layer_dropout[0])
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        for l in range(self.args.layers):

            x = (self.linear_array.linears[l])(x)
            if l < self.args.layers-1 :
                x = F.relu(x)

        """
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        """

        output = F.log_softmax(x, dim=1)

        return output