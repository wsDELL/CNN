import multiprocessing
from multiprocessing import Queue
from multiprocessing.managers import BaseManager

from Model import AlexNet
from Model.vgg import *
from Model.dis_alexnet import *

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
    datadict = unpickle(filename)  # dict类型
    X = datadict[b'data']  # X, ndarray, 像素值
    Y = datadict[b'labels']  # Y, list, 标签, 分类

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


def LoadData(num_worker):
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
    mdr.total_Shuffle()
    mdr.GenerateValidationSet(k=12)
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    return mdr


def LoadData1():
    mdr = MnistDataReader(train_x, train_y, test_x, test_y, "image")
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr


task_queue = Queue()
result_queue = Queue()
total_order_queue = Queue()
training_order_queue = Queue()
check_order_status_queue = Queue()
send_order_status_queue = Queue()


def return_total_order_queue():
    global total_order_queue
    return total_order_queue


def return_training_order_queue():
    global training_order_queue
    return training_order_queue


def return_check_status_queue():
    global check_order_status_queue
    return check_order_status_queue


def return_send_order_status_queue():
    global send_order_status_queue
    return send_order_status_queue


def return_task_queue():
    global task_queue
    return task_queue


def return_result_queue():
    global result_queue
    return result_queue


class QueueManager(BaseManager):
    pass


if __name__ == '__main__':
    time1 = time.time()
    num_output = 10
    max_epoch = 100
    batch_size = 128
    learning_rate = 0.005
    hp = HyperParameters(learning_rate, max_epoch, batch_size, net_type=NetType.MultipleClassifier,
                         optimizer_name=OptimizerName.Adam, regular_name=RegularMethod.L2, regular_value=0.0005)
    lock = multiprocessing.Lock()
    num_worker = 2
    dataReader = LoadData(num_worker)
    # net = model()
    net = AlexNet(param=hp, model_name="dis_Alexnet")
    # net = VGG(param=params, vgg_name="VGG11")
    param = net.distributed_save_parameters()
    net.loss_func = LossFunction(net.hp.net_type)
    if net.hp.regular_name == RegularMethod.EarlyStop:
        net.loss_trace = TrainingHistory(True, net.hp.regular_value)
    else:
        net.loss_trace = TrainingHistory()

    if net.hp.batch_size == -1 or net.hp.batch_size > dataReader.num_train:
        net.hp.batch_size = dataReader.num_train
    checkpoint = 0.1
    max_iteration = math.ceil(dataReader.num_train / net.hp.batch_size)
    checkpoint_iteration = int(math.ceil(max_iteration * checkpoint))
    need_stop = False
    QueueManager.register('get_task_queue', callable=return_task_queue)
    QueueManager.register('get_gradient_queue', callable=return_result_queue)
    QueueManager.register('send_order_queue', callable=return_total_order_queue)
    QueueManager.register('send_training_order_queue', callable=return_training_order_queue)
    QueueManager.register('send_data_status', callable=return_send_order_status_queue)
    QueueManager.register('check_data_status', callable=return_check_status_queue)

    manager = QueueManager(address=('131.181.249.163', 5006), authkey=b'abc')
    manager.start()

    total_order = manager.send_order_queue()
    training_order = manager.send_training_order_queue()
    status = manager.send_data_status()
    checking = manager.check_data_status()
    task = manager.get_task_queue()
    result = manager.get_gradient_queue()

    data_status_set = []
    total_iteration_count = 0
    valid_count = 0
    data_order_set = dataReader.data_split(num_worker)

    for i in range(num_worker):
        total_order.put(data_order_set[i])

    # while True:
    #     data_status_set.append(order_status.get())
    #     if len(data_status_set) == num_worker:
    #         break

    # training_order = order_status
    for epoch in range(net.hp.max_epoch):
        iteration_count = 0
        epoch_status = 0
        while True:
            print('put task %d' % iteration_count)
            task.put({iteration_count: param, "epoch_status": 0})
            iteration_count = iteration_count + 1
            if iteration_count % num_worker == 0:
                lock.acquire()
                grads = []
                while True:
                    grad = result.get()
                    grads.append(grad)
                    if len(grads) == num_worker:
                        break
                for i in range(len(grads)):
                    if i == 0:
                        net.distributed_load_gradient(grads[i])
                        epoch_status += grads[i]['epoch_status']
                        # net.update()
                    else:
                        net.distributed_add_gradient(grads[i])
                        epoch_status += grads[i]['epoch_status']
                net.distributed_average_gradient(num_worker)
                net.update()
                batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, iteration_count)
                # net.accuracy_cal(batch_x, batch_y, epoch, iteration_count)
                param = net.distributed_save_parameters()
                lock.release()
                print('update finish')
            total_iteration_count += 1
            if iteration_count % checkpoint_iteration == 0:
                if iteration_count > max_iteration:
                    iteration_count = max_iteration - 1
                batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, iteration_count)
                if batch_x.shape[0] < net.hp.batch_size:
                    batch_x, batch_y = dataReader.GetBatchTrainSamples(net.hp.batch_size, iteration_count - 1)
                need_stop = net.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration_count)
                net.SaveLossHistory(valid_count, name="dis_alexnet4.csv")
                valid_count += 1
                if need_stop:
                    break

            if epoch_status == num_worker:
                break
        net.save_parameters()
    print("testing...")
    accuracy = net.Test(dataReader)
    print(accuracy)
    time2 = time.time()
    print(f"total time: {time2 - time1}")
    manager.shutdown()
    print("master exit.")
