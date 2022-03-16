import random
import multiprocessing
import sys
from multiprocessing.managers import BaseManager
from multiprocessing import Queue
import numpy as np
from MiniFramework import *
from Model.vgg import *

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


def LoadData1():
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


    # c1 = ConLayer((1, 28, 28), (8, 3, 3), params, stride=1, padding=0)
    # net.add_layer(c1, "c1")
    # r1 = ActivationLayer(ReLU())
    # net.add_layer(r1, "relu1")
    # p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    # net.add_layer(p1, "p1")
    #
    # c2 = ConLayer(p1.output_shape, (16, 3, 3), params, stride=1, pad=0)
    # net.add_layer(c2, "c2")
    # r2 = ActivationLayer(ReLU())
    # net.add_layer(r2, "relu2")
    # p2 = PoolingLayer(c2.output_shape, (2, 2), 2, PoolingTypes.MAX)
    # net.add_layer(p2, "p2")
    #
    # f3 = FCLayer(p2.output_size, 32, params)
    # net.add_layer(f3, "f3")
    # bn3 = BatchNormalLayer(f3.output_num)
    # net.add_layer(bn3, "bn3")
    # r3 = ActivationLayer(ReLU())
    # net.add_layer(r3, "relu3")
    #
    # f4 = FCLayer(f3.output_num, 10, params)
    # net.add_layer(f4, "f2")
    # s4 = ClassificationLayer(Softmax())
    # net.add_layer(s4, "s4")

    return net


def model1():
    num_output = 10
    max_epoch = 20
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters(learning_rate, max_epoch, batch_size,
                             net_type=NetType.MultipleClassifier,
                             init_method=InitialMethod.Xavier,
                             optimizer_name=OptimizerName.SGD)

    net = NeuralNet(params,"alexnet")

    c1 = ConLayer((1, 28, 28), (32, 3, 3), params, stride=1, padding=1)
    net.add_layer(c1, "c1")
    p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1")
    r1 = ActivationLayer(ReLU())
    net.add_layer(r1, "relu1")

    c2 = ConLayer(p1.output_shape, (64, 3, 3), params, stride=1, padding=1)
    net.add_layer(c2, "c2")
    p2 = PoolingLayer(c2.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")
    r2 = ActivationLayer(ReLU())
    net.add_layer(r2, "relu2")

    c3 = ConLayer(p2.output_shape, (128, 3, 3), params, stride=1, padding=1)
    net.add_layer(c3, "c3")
    # r3 = ActivationLayer(ReLU())
    # net.add_layer(r3, "relu3")

    c4 = ConLayer(c3.output_shape, (256, 3, 3), params, stride=1, padding=1)
    net.add_layer(c4, "c3")
    # r4 = ActivationLayer(ReLU())
    # net.add_layer(r4, "relu4")

    c5 = ConLayer(c4.output_shape, (256, 3, 3), params, stride=1, padding=1)
    net.add_layer(c5, "c3")
    p5 = PoolingLayer(c5.output_shape, (3, 3), 2, PoolingTypes.MAX)
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


task_queue = Queue()
result_queue = Queue()


def return_task_queue():
    global task_queue
    return task_queue


def return_result_queue():
    global result_queue
    return result_queue


class QueueManager(BaseManager):
    pass
    # def get_task_queue(self):
    #     pass
    #
    # def get_result_queue(self):
    #     pass


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
    lock = multiprocessing.Lock()
    dataReader = LoadData()
    # net = model()
    net = VGG(param=params,vgg_name="VGG11")
    param = net.distributed_save_parameters()
    # net.train(dataReader, checkpoint=0.05, need_test=True)
    net.loss_func = LossFunction(net.hp.net_type)
    if net.hp.regular_name == RegularMethod.EarlyStop:
        net.loss_trace = TrainingHistory(True, net.hp.regular_value)
    else:
        net.loss_trace = TrainingHistory()

    if net.hp.batch_size == -1 or net.hp.batch_size > dataReader.num_train:
        net.hp.batch_size = dataReader.num_train
    checkpoint = 0.05
    max_iteration = math.ceil(dataReader.num_train / net.hp.batch_size)
    checkpoint_iteration = int(math.ceil(max_iteration * checkpoint))
    need_stop = False
    QueueManager.register('get_task_queue', callable=return_task_queue)
    QueueManager.register('get_result_queue', callable=return_result_queue)

    manager = QueueManager(address=('131.181.249.163', 5006), authkey=b'abc')
    manager.start()
    # s = manager.get_server()
    # s.serve_forever()

    task = manager.get_task_queue()
    result = manager.get_result_queue()
    result1 = []
    for epoch in range(net.hp.max_epoch):
        print(f"epoch {epoch} start")
        # dataReader.Shuffle()
        iteration_count = 0
        for iteration in range(max_iteration):
            print('put task %d' % iteration)
            if iteration == 2:
                print("stop")
            task.put({iteration: param})
            iteration_count = iteration_count + 1

            if iteration_count % 5 == 0:
                lock.acquire()
                ret = result.get()
                result1.append(ret)
                net.distributed_load_parameters(result1[0])
                net.distributed_add_parameters(result1[1])
                net.distributed_add_parameters(result1[2])
                net.distributed_add_parameters(result1[3])
                net.distributed_add_parameters(result1[4])
                net.distributed_average_parameters(5)
                param = net.distributed_save_parameters()
                lock.release()
                print('update finish')
                result1 = []
                iteration_count = 0
            batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, iteration)
            total_iteration = epoch * max_iteration + iteration

            # if (total_iteration + 1) % checkpoint_iteration == 0:
            #     need_stop = net.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
            #     if need_stop:
            #         break
            # name = list(ret.keys())[0]
            # print(name)
    print("testing...")
    accuracy = net.Test(dataReader)
    print(accuracy)
    time2 = time.time()
    print(f"total time: {time2 - time1}")


        # while True:



    # nums = picture_extract.num_of_mnist_data(data_dir)
    # images = picture_extract.images_of_mnist_data(data_dir)
    # image0 = images[0]
    # image1 = images[0:160]
    # print(sys.getsizeof(image1))
    # print(sys.getsizeof(images[1]))
    # print(sys.getsizeof(images[2]))
    # print(sys.getsizeof(images[3]))
    # print(sys.getsizeof(images[4]))


    # nums = [x for x in range(nums)]
    # nums = np.array_split(nums, 1000)
    # for i in range(len(nums)):
    #     print('put task %d' % i)
    #     task.put(nums[i])
    # while True:
    #     if (result.qsize() == len(nums)):
    #         for i in range(len(nums)):
    #             ret = result.get()
    #             print(f'result {i}: {ret}')
    #         break

    manager.shutdown()
    print("master exit.")
