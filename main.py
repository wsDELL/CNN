# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import MiniFramework
from MiniFramework import *
import os
import time

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


def LoadData():
    mdr = MnistDataReader(train_x, train_y, test_x, test_y, "image")
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
    r1 = ActivationLayer(ReLU())
    net.add_layer(r1, "relu1")

    c2 = ConLayer(32, 64, kernel_size=3, hp=params, stride=1)
    net.add_layer(c2, "c2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")
    r2 = ActivationLayer(ReLU())
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
    r5 = ActivationLayer(ReLU())
    net.add_layer(r5, "relu5")

    # d1 = DropoutLayer(p5.output_shape)
    # net.add_layer(d1, 'd1')
    f1 = FCLayer(256 * 3 * 3, 1024, params)
    net.add_layer(f1, "f1")
    bn1 = BatchNormalLayer(f1.output_num)
    net.add_layer(bn1, 'bn1')
    r6 = ActivationLayer(ReLU())
    net.add_layer(r6, "relu6")

    # d1 = DropoutLayer(f1.output_num)
    # net.add_layer(d1, "d1")

    f2 = FCLayer(f1.output_num, 512, params)
    net.add_layer(f2, "f2")
    bn2 = BatchNormalLayer(f2.output_num)
    net.add_layer(bn2, 'bn1')
    r6 = ActivationLayer(ReLU())
    net.add_layer(r6, "relu6")

    f3 = FCLayer(f2.output_num, 10, params)
    net.add_layer(f3, "f3")
    s4 = ClassificationLayer(Softmax())
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
    r1 = ActivationLayer(ReLU())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConLayer(8, 16, kernel_size=3, hp=params, stride=1)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(ReLU())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(kernel_size=2, stride=2, pooling_type=PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    f3 = FCLayer(400, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BatchNormalLayer(32)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(ReLU())
    net.add_layer(r3, "relu3")

    f4 = FCLayer(32, 10, params)
    net.add_layer(f4, "f2")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net


if __name__ == '__main__':
    time1 = time.time()
    dataReader = LoadData()
    net = model()
    # net.distributed_load_parameters()
    net.train(dataReader, checkpoint=0.05, need_test=True)
    time2 = time.time()
    print(f"total time: {time2 - time1}")
    # checkpoint = 0.05
    # max_iteration = math.ceil(dataReader.num_train / net.hp.batch_size)
    # checkpoint_iteration = int(math.ceil(max_iteration * checkpoint))
    # need_stop = False
    # for epoch in range(net.hp.max_epoch):
    #     dataReader.Shuffle()
    #
    #     for iteration in range(max_iteration):
    #         # get x and y value for one sample
    #         batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, iteration)
    #         # for optimizers which need pre-update weights
    #         # if self.hp.optimizer_name == OptimizerName.Nag:
    #         #     self.__pre_update()
    #         # get z from x,y
    #
    #         time1 = time.time()
    #         self.__forward(batch_x, train=True)
    #         time2 = time.time()
    #         # calculate gradient of w and b
    #         self.__backward(batch_x, batch_y)
    #         time3 = time.time()
    #         # final update w,b
    #         self.__update()
    #         time4 = time.time()
    #         print(f"iteration {iteration} , forward time: {time2 - time1}, "
    #               f"backward time: {time3 - time2}, update time: {time4 - time3},total time: {time4 - time1}")
    #
    #         total_iteration = epoch * max_iteration + iteration
    #         if (total_iteration + 1) % checkpoint_iteration == 0:
    #             need_stop = self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
    #             if need_stop:
    #                 break
    #
    #     self.save_parameters()
    #     if need_stop:
    #         break
    #     # end if
    # # end for
    #
    # t1 = time.time()
    # print("time used:", t1 - t0)
    #
    #
    # accuracy = net.Test(dataReader)
    net.ShowLossHistory(XCoordinate.Iteration)
