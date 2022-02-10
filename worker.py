import time
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Queue, Array

from MiniFramework import *


train_x = "./data/MNIST/raw/train-images-idx3-ubyte"
train_y = "./data/MNIST/raw/train-labels-idx1-ubyte"
test_x = "./data/MNIST/raw/t10k-images-idx3-ubyte"
test_y = "./data/MNIST/raw/t10k-labels-idx1-ubyte"

class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

def LoadData():
    mdr = MnistDataReader(train_x, train_y, test_x, test_y, "image")
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr


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

    c1 = ConLayer((1, 28, 28), (8, 3, 3), params, stride=1, pad=0)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(ReLU())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConLayer(p1.output_shape, (16, 3, 3), params, stride=1, pad=0)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(ReLU())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    f3 = FCLayer(p2.output_size, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BatchNormalLayer(f3.output_num)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(ReLU())
    net.add_layer(r3, "relu3")

    f4 = FCLayer(f3.output_num, 10, params)
    net.add_layer(f4, "f2")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net
if __name__ == '__main__':


    server_address = '131.181.249.244'
    print('connect to server %s...' % server_address)

    manager = QueueManager(address=(server_address, 5006), authkey=b'abc')
    manager.connect()

    task = manager.get_task_queue()
    result = manager.get_result_queue()

    while True:
        one_res = []
        if task.empty():
            break
        n = task.get()
        print(f'worker get {n}')
        for i in n:
            one_res.append(labels[i])
        result.put(one_res)

    print('worker exit')
