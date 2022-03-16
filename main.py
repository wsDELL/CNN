# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import MiniFramework
from MiniFramework import *
import os
import time
from Model.vgg import *

# class AlexNet(NeuralNet):
#     def __init__(self, params, model_name):
#         super().__init__(params, model_name)
#         self.add_layer(MiniFramework.ConLayer(input_shape=(1,28,28), kernal_shape=(32,3,3), hp=params, pad=1))
#         self.add_layer(MiniFramework.PoolingLayer(layer_type="Pool", input_shape=(32,14,14),pool_shape=(2,2),stride=2))
#         self.add_layer(MiniFramework.ReLU())
#         self.add_layer(MiniFramework.ConLayer(input_shape=(32,14,14),kernal_shape=(64,3,3),hp=params,pad=1))
#         self.add_layer(MiniFramework.PoolingLayer(input_shape=(64,7,7),pool_shape=(64,3,3),stride=2,layer_type='Pool'))
#         self.add_layer(MiniFramework.ReLU())
#         self.add_layer(MiniFramework.ConLayer(input_shape=(64,7,7), kernal_shape=(128,3,3), hp=params, pad=1))
#         self.add_layer(MiniFramework.ConLayer(input_shape=(128,7,7), kernal_shape=(256,3,3), hp=params, pad=1))
#         self.add_layer(MiniFramework.ConLayer(input_shape=(256,7,7), kernal_shape=(256,3,3), hp=params, pad=1))
#         self.add_layer(MiniFramework.PoolingLayer(layer_type="Pool", input_shape=(256,7,7),pool_shape=(3,3),stride=2))
#         self.add_layer(MiniFramework.ReLU())
#         self.add_layer(MiniFramework.FCLayer(256*3*3,1024))
#
#
#
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     hp = MiniFramework.HyperParameters()
#     con1 = ConLayer(input_shape=(1, 28, 28), kernal_shape=(1, 3, 3), hp=hp, layer_type="CON", stride=2, pad=1)
#     pool1 = MiniFramework.PoolingLayer(layer_type="Pool", input_shape=(1, 28, 28), pool_shape=(2, 2), stride=2)
#     batch1 = MiniFramework.BacthNormalLayer(layer_type='batchlayer',input_size=28)
#     activator = MiniFramework.ReLU()
#     liner1 = MiniFramework.Softmax()
#     drop1 = MiniFramework.DropoutLayer(layer_type='dropout', input_size=2)
#     fclayer1 = MiniFramework.FCLayer(input_n=1*28*28,output_n=196,hp=hp)

#     dataReader=MnistDataReader(train_x,train_y,test_x,test_y)
#     dataReader.ReadData()
#     train_X = dataReader.XTrainRaw
#     oner = train_X[0]
#     oner = oner[np.newaxis,:]
#     res = con1.forward(oner)
#     res2 = pool1.forward(oner)
#     res3 = batch1.forward(oner)
#     res4 = activator.forward(oner)
#     oner1 = np.transpose(oner).reshape(1*28*28,1)
#     res6 = drop1.forward(oner1)
#     res5 = liner1.forward(oner1)
#
#     print(res.shape)
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
train_x = "./data/MNIST/raw/train-images-idx3-ubyte"
train_y = "./data/MNIST/raw/train-labels-idx1-ubyte"
test_x = "./data/MNIST/raw/t10k-images-idx3-ubyte"
test_y = "./data/MNIST/raw/t10k-labels-idx1-ubyte"

cifar_name = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_CIFAR_batch(filename):
    # """ load single batch of cifar """
    # # with open(filename, 'rb') as f:
        datadict = unpickle(filename)  # dict类型
        X = datadict[b'data']  # X, ndarray, 像素值
        Y = datadict[b'labels']  # Y, list, 标签, 分类

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def LoadData():
    for i in range(len(cifar_name)):
        if i == 0:
            train_x,train_y=load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
        else:
            train_X,train_Y=load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/{cifar_name[i]}")
            train_x = np.concatenate((train_x,train_X))
            train_y = np.concatenate((train_y,train_Y))
    test_x,test_y = load_CIFAR_batch(filename=f"./data/cifar-10-batches-py/test_batch")

    mdr = CIFAR10DataReader(train_x, train_y, test_x, test_y)
    # mdr = MnistDataReader(train_x,train_y,test_x,test_y)
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr


def model1():
    num_output = 10
    max_epoch = 20
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(learning_rate, max_epoch, batch_size,
                             net_type=NetType.MultipleClassifier,
                             init_method=InitialMethod.Xavier,
                             optimizer_name=OptimizerName.SGD)

    net = NeuralNet(params, "alexnet")

    c1 = ConLayer(1, 32, kernel_size=3, hp=params, stride=1)
    net.add_layer(c1, "c1")
    p1 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p1, "p1")
    r1 = ReLU()
    net.add_layer(r1, "relu1")

    c2 = ConLayer(32, 64, kernel_size=3, hp=params, stride=1)
    net.add_layer(c2, "c2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")
    r2 = ReLU()
    net.add_layer(r2, "relu2")

    c3 = ConLayer(64, 128, kernel_size=3, hp=params, stride=1)
    net.add_layer(c3, "c3")
    # r3 = ActivationLayer(ReLU())
    # net.add_layer(r3, "relu3")

    c4 = ConLayer(128, 256, kernel_size=3, hp=params, stride=1)
    net.add_layer(c4, "c3")
    # r4 = ActivationLayer(ReLU())
    # net.add_layer(r4, "relu4")

    c5 = ConLayer(256, 256, kernel_size=3, hp=params, stride=1)
    net.add_layer(c5, "c3")
    p5 = PoolingLayer(kernel_size=3, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p5, "p2")
    r5 = ReLU()
    net.add_layer(r5, "relu5")

    # d1 = DropoutLayer(p5.output_shape)
    # net.add_layer(d1, 'd1')
    f1 = FCLayer(256 * 3 * 3, 1024, params)
    net.add_layer(f1, "f1")
    bn1 = BatchNormalLayer(f1.output_num)
    net.add_layer(bn1, 'bn1')
    r6 = ReLU()
    net.add_layer(r6, "relu6")

    # d1 = DropoutLayer(f1.output_num)
    # net.add_layer(d1, "d1")

    f2 = FCLayer(f1.output_num, 512, params)
    net.add_layer(f2, "f2")
    bn2 = BatchNormalLayer(f2.output_num)
    net.add_layer(bn2, 'bn1')
    r6 = ReLU()
    net.add_layer(r6, "relu6")

    f3 = FCLayer(f2.output_num, 10, params)
    net.add_layer(f3, "f3")
    s4 = Softmax()
    net.add_layer(s4, "s4")

    return net


def model():
    num_output = 10
    max_epoch = 5
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet(params, "mnist_cnn")

    c1 = ConLayer(1, 8, kernel_size=3, hp=params, stride=1)
    net.add_layer(c1, "c1")
    r1 = ReLU()
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConLayer(8, 16, kernel_size=3, hp=params, stride=1)
    net.add_layer(c2, "c2")
    r2 = ReLU()
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    f3 = FCLayer(400, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BatchNormalLayer(32)
    net.add_layer(bn3, "bn3")
    r3 = ReLU()
    net.add_layer(r3, "relu3")

    f4 = FCLayer(32, 10, params)
    net.add_layer(f4, "f2")
    s4 = Softmax()
    net.add_layer(s4, "s4")

    return net


if __name__ == '__main__':
    time1 = time.time()
    num_output = 10
    max_epoch = 5
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Adam)
    dataReader = LoadData()
    # net = model1()
    # net.distributed_load_parameters()
    net = VGG(param=params,vgg_name="VGG11")
    print("start")
    net.train(dataReader, checkpoint=0.05, need_test=True)
    print("end")
    time2 = time.time()
    print(f"total time: {time2 - time1}")
    net.ShowLossHistory(XCoordinate.Iteration)
