# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import MiniFramework
from MiniFramework import *
import os
import time
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
    mdr.Shuffle()
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


def Alexnet():
    num_output = 10
    max_epoch = 40
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Kaiming_Normal,
        optimizer_name=OptimizerName.Adam, regular_name=RegularMethod.L2, regular_value=0.0005)

    net = NeuralNet(params, "alexnet")

    c1 = ConLayer(3, 64, kernel_size=3, hp=params, stride=2, padding=1)
    net.add_layer(c1, "c1")
    r1 = ReLU()
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConLayer(64, 192, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c2, "c2")
    r2 = ReLU()
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    c3 = ConLayer(192, 384, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c3, "c3")
    r3 = ReLU()
    net.add_layer(r3, "relu3")

    c4 = ConLayer(384, 256, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c4, "c4")
    r4 = ReLU()
    net.add_layer(r4, "relu4")

    c5 = ConLayer(256, 256, kernel_size=3, hp=params, stride=1, padding=1)
    net.add_layer(c5, "c5")
    r5 = ReLU()
    net.add_layer(r5, "relu5")
    p5 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MEAN)
    net.add_layer(p5, "p5")

    d1 = DropoutLayer(ratio=0.3)
    net.add_layer(d1, 'd1')
    f1 = FCLayer(256 * 2 * 2, 1024, params)
    net.add_layer(f1, "f1")
    bn1 = BatchNormalLayer(f1.output_num)
    net.add_layer(bn1, 'bn1')
    r6 = ReLU()
    net.add_layer(r6, "relu6")

    d2 = DropoutLayer(ratio=0.3)
    net.add_layer(d2, "d2")
    f2 = FCLayer(1024, 1024, params)
    net.add_layer(f2, "f2")
    bn2 = BatchNormalLayer(f2.output_num)
    net.add_layer(bn2, 'bn2')
    r6 = ReLU()
    net.add_layer(r6, "relu6")

    f3 = FCLayer(f2.output_num, 10, params)
    net.add_layer(f3, "f3")
    s4 = Softmax()
    net.add_layer(s4, "s4")

    return net


if __name__ == '__main__':
    time1 = time.time()
    num_output = 10
    max_epoch = 20
    batch_size = 128
    learning_rate = 0.005
    dataReader = LoadData()
    net = Alexnet()
    print("start")
    net.train(dataReader, checkpoint=1, need_test=True, file_name="alexnet_loss_data2.csv")
    print("end")
    time2 = time.time()
    print(f"total time: {time2 - time1}")
    # net.ShowLossHistory(XCoordinate.Iteration)
