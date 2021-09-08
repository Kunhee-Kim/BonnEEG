

#import

import sys
import os
import pandas
import torch
import numpy as np

import Bonn_dataset.F as f_data

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import torchvision

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.model_selection import KFold
import argparse
import json

import argparse
import json

import Bonn_directory.model as Model
from Bonn_directory import training
from Bonn_directory import validation
from Bonn_directory import testing
from Bonn_directory import dataload


# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed_all(777)


# params
learning_rate = 0.001
training_epochs = 10
batch_size = 64
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

freq = 173.61


# default params
layers = 2
"""
layer_features = np.ones(layers)
layer_dropout = np.zeros(layers)
"""
layer_features = np.array([128, 10])
layer_dropout = np.array([0.25, 0.5])  # dropout = 0 if not used

splits = 4


# argument parse


with open('argset1.json', 'r') as default_json1:
    default_args1 = json.load(default_json1)


parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=default_args1.get('layers'))
parser.add_argument('--layer_features', type=int, nargs='*', default=default_args1.get('layer_features'))
parser.add_argument('--layer_dropout', type=float, nargs='*', default=default_args1.get('layer_dropout'))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--training_epochs', type=int, default=8)
parser.add_argument('--test', type=int, default=default_args1.get('test'))

args = parser.parse_args()
args.layer_features = np.array(args.layer_features)
args.layer_dropout = np.array(args.layer_dropout)
print('epochs :', args.training_epochs)
print('features :', args.layer_features, type(args.layer_features))
print('dropout :', args.layer_dropout, type(args.layer_dropout))


# writer

writer = SummaryWriter(log_dir='./runs')


# main
def main(args):
    kfold = KFold(n_splits=splits, shuffle=True, random_state=True)
    dataloader = dataload.dataloader(args)
    model_final = Model.CNN(args).double()
    max_val_acc = 0
    train_loss_write = 0

    for train_index, val_index in kfold.split(dataloader.training_set):

        # k-fold split
        model = Model.CNN(args).double()
        train_sub_Bonn = torch.utils.data.Subset(dataloader.training_set, train_index)
        val_sub_Bonn = torch.utils.data.Subset(dataloader.training_set, val_index)
        train_sub_dataloader = DataLoader(train_sub_Bonn, batch_size=batch_size, shuffle=True, drop_last=True)
        val_sub_dataloader = DataLoader(val_sub_Bonn, batch_size=batch_size, shuffle=True, drop_last=True)

        # train, validate
        training_loss = training.training(train_sub_dataloader, model, args)
        validation_loss, validation_accuracy = validation.validation(val_sub_dataloader, model, args)

        if validation_loss > max_val_acc :
            max_val_acc = validation_accuracy
            train_loss_write = training_loss
            model_final = model

    writer.add_scalar('train_loss/epochs', train_loss_write, args.training_epochs)
    writer.add_scalar('validation_accuracy/epochs', max_val_acc, args.training_epochs)

    # testing
    print('test len :', len(dataloader.test_set))
    _, test_accuracy = testing.testing(dataloader.test_generator, model_final, args)
    writer.add_scalar('test_accuracy/epochs', test_accuracy, args.training_epochs)
    writer.add_scalar('test_accuracy/layers', test_accuracy, args.layers)
    writer.add_scalar('test_accuracy/layer_feature[0', test_accuracy, args.layer_features[0])

    writer.close()

    return None


if __name__ == '__main__':
    main(args)

