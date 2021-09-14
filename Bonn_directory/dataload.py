
# import

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
import augmentation as aug

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.model_selection import KFold
import argparse
import json

import argparse
import json


# params

params = {'shuffle': True,
          'num_workers': 6}

# get files
directory_F = '/Users/kimgunhee/workspace/BonnEEG/Bonn_dataset/F'
directory_N = '/Users/kimgunhee/workspace/BonnEEG/Bonn_dataset/N'
directory_O = '/Users/kimgunhee/workspace/BonnEEG/Bonn_dataset/O'
directory_S = '/Users/kimgunhee/workspace/BonnEEG/Bonn_dataset/S'
directory_Z = '/Users/kimgunhee/workspace/BonnEEG/Bonn_dataset/Z'

fileList_F = os.listdir(directory_F)
fileList_N = os.listdir(directory_N)
fileList_O = os.listdir(directory_O)
fileList_S = os.listdir(directory_S)
fileList_Z = os.listdir(directory_Z)

# get raw data
rawdata_F = list([])
for f in fileList_F :
    temp_arr = []
    r = open(directory_F +'/'+ f, mode = 'r')
    for lines in r.readlines():
        temp_arr.append(lines.splitlines()[0])
    rawdata_F.append(np.array(temp_arr, dtype='double'))

rawdata_N = list([])
for f in fileList_N :
    temp_arr = []
    r = open(directory_N +'/'+ f, mode = 'r')
    for lines in r.readlines():
        temp_arr.append(lines.splitlines()[0])
    rawdata_N.append(np.array(temp_arr, dtype='double'))

rawdata_O = list([])
for f in fileList_O :
    temp_arr = []
    r = open(directory_O +'/'+ f, mode = 'r')
    for lines in r.readlines():
        temp_arr.append(lines.splitlines()[0])
    rawdata_O.append(np.array(temp_arr, dtype='double'))

rawdata_S = list([])
for f in fileList_S :
    temp_arr = []
    r = open(directory_S +'/'+ f, mode = 'r')
    for lines in r.readlines():
        temp_arr.append(lines.splitlines()[0])
    rawdata_S.append(np.array(temp_arr, dtype='double'))

rawdata_Z = list([])
for f in fileList_Z :
    temp_arr = []
    r = open(directory_Z +'/'+ f, mode = 'r')
    for lines in r.readlines():
        temp_arr.append(lines.splitlines()[0])
    rawdata_Z.append(np.array(temp_arr, dtype='double'))

total_raw_data = list([])
total_raw_data+=rawdata_F
total_raw_data+=rawdata_N
total_raw_data+=rawdata_O
total_raw_data+=rawdata_S
total_raw_data+=rawdata_S ### add S twice to make frequency of all labels equal
total_raw_data+=rawdata_Z


# define labels

labels_F = [[0,0,1]]*100
labels_N = [[0,0,1]]*100
labels_O = [[0,1,0]]*100
labels_S = [[1,0,0]]*100
labels_Z = [[0,1,0]]*100

total_labels = list([])
total_labels+=labels_F
total_labels+=labels_N
total_labels+=labels_O
total_labels+=labels_S
total_labels+=labels_S ### add S twice to make frequency of all labels equal
total_labels+=labels_Z
total_labels = np.array(total_labels)


# training, test set
import numpy as np

total_index = np.arange(0, 499)
np.random.shuffle(total_index)
test_index = total_index[:100]
training_index = total_index[100:]

training_raw_data = list([])
training_labels = list([])
test_raw_data = list([])
test_labels = list([])

for i in training_index :
    training_raw_data.append(total_raw_data[i])
    training_labels.append(total_labels[i])

    training_raw_data.append(aug.jitter(total_raw_data[i]))
    training_labels.append(aug.same(total_labels[i]))
    training_raw_data.append(aug.scaling(total_raw_data[i]))
    training_labels.append(aug.same(total_labels[i]))
    training_raw_data.append(aug.rotation(total_raw_data[i]))
    training_labels.append(aug.same(total_labels[i]))
    training_raw_data.append(aug.permutation(total_raw_data[i]))
    training_labels.append(aug.same(total_labels[i]))

for i in test_index :
    test_raw_data.append(total_raw_data[i])
    test_labels.append(total_labels[i])


# class Dataset(X, y)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        'Initialization'
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = np.array([self.dataset[index]], dtype = 'double')
        y = self.labels[index]

        return X, y


# generator

class dataloader():
    def __init__(self, args):
        self.args = args
        self.transform = torchvision.transforms.ToTensor()

        """
        training_set = Dataset(training_raw_data, training_labels)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        test_set = Dataset(test_raw_data, test_labels)
        test_generator = torch.utils.data.DataLoader(test_set, **params)
        """

        self.training_set = Dataset(training_raw_data, training_labels)
        self.training_generator = torch.utils.data.DataLoader(self.training_set, **params, batch_size=args.batch_size)
        self.test_set = Dataset(test_raw_data, test_labels)
        self.test_generator = torch.utils.data.DataLoader(self.test_set, **params, batch_size=args.batch_size)


"""
class dataloader():

    def __init__(self, args):
        self.args = args
        transform = transforms.ToTensor()
        self.train_mnist = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_mnist = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.train_dataloader = DataLoader(self.train_mnist, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_mnist, batch_size=args.batch_size, shuffle=True, drop_last=True)
"""