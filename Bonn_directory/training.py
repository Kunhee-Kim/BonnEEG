import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


def training(tr_generator, cnn_model, args):

    training_batch = len(tr_generator)
    tr_criterion = nn.CrossEntropyLoss()
    tr_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=args.learning_rate)
    loss = 0

    for epoch in range(args.training_epochs):
        avg_loss = 0

        for inputs, labels in tr_generator:

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            labels_max = torch.max(labels, 1)[1]

            tr_optimizer.zero_grad()

            outputs = cnn_model(inputs)
            loss = tr_criterion(outputs, torch.max(labels,1)[1])
            loss.backward()
            tr_optimizer.step()

            avg_loss += loss/300

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_loss))
        loss = avg_loss

    return loss
