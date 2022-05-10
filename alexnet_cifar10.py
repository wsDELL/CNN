# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import MiniFramework
from MiniFramework import *
import os
import time

from Model.alexnet import AlexNet
from Model.vgg import *

train_x = "./data/MNIST/raw/train-images-idx3-ubyte"
train_y = "./data/MNIST/raw/train-labels-idx1-ubyte"
test_x = "./data/MNIST/raw/t10k-images-idx3-ubyte"
test_y = "./data/MNIST/raw/t10k-labels-idx1-ubyte"

cifar_name = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR_batch(filename):
    # """ load single batch of cifar """
    # # with open(filename, 'rb') as f:
    datadict = unpickle(filename)
    X = datadict[b'data']
    Y = datadict[b'labels']

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


def LoadData():
    for i in range(len(cifar_name)):
        if i == 0:
            train_x, train_y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
        else:
            train_X, train_Y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
            train_x = np.concatenate((train_x, train_X))
            train_y = np.concatenate((train_y, train_Y))
    test_x, test_y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/test_batch")

    mdr = CIFAR10DataReader(train_x, train_y, test_x, test_y)
    # mdr = MnistDataReader(train_x,train_y,test_x,test_y)
    mdr.ReadData()
    mdr.training_Shuffle()
    mdr.GenerateValidationSet(k=12)
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    return mdr


def LoadData1():
    mdr = MnistDataReader(train_x, train_y, test_x, test_y)
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr

if __name__ == '__main__':
    time1 = time.time()
    num_output = 10
    max_epoch = 50
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters(learning_rate, max_epoch, batch_size, net_type=NetType.MultipleClassifier,
                             optimizer_name=OptimizerName.Adam, regular_name=RegularMethod.L2, regular_value=0.0005)
    dataReader = LoadData()
    net = AlexNet(param=params, model_name="Alexnet")
    print("start")
    net.train(dataReader, checkpoint=0.1, need_test=True, file_name="alexnet_loss_data5.csv")
    print("end")
    time2 = time.time()
    print(f"total time: {time2 - time1}")
    # net.ShowLossHistory(XCoordinate.Iteration)
