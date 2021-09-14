import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


def validation(te_dataloader, cnn_model, args):
    te_avg_loss = 0
    te_batch = len(te_dataloader)
    te_total = 0
    te_correct = 0

    tr_criterion = nn.CrossEntropyLoss()
    tr_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=args.learning_rate)

    with torch.no_grad():
        for inputs_te, labels_te in te_dataloader:

            inputs_te = inputs_te.to(args.device)
            labels_te = labels_te.to(args.device)
            outputs_te = cnn_model(inputs_te)

            labels_te_max = torch.max(labels_te, 1)[1]
            loss_te = tr_criterion(outputs_te, labels_te_max)
            te_avg_loss+= loss_te/320

            _, predicted = torch.max(outputs_te.data, 1)
            for label, prediction in zip(labels_te_max, predicted):
                if label == prediction:
                    te_correct += 1
                te_total += 1

    te_accuracy = 0
    if te_total != 0:
        te_accuracy = te_correct / te_total
    print('validation loss : %f' % te_avg_loss)
    print('validation accuracy : %f' % te_accuracy)

    return te_avg_loss, te_accuracy
