import math

from MiniFramework import *
from MiniFramework import FCLayer, ConLayer
from MiniFramework.LossFunction import *
from MiniFramework.TrainingHistory import *
import os
import time
import csv


class NeuralNet(object):
    def __init__(self, params, model_name):
        self.loss_trace = None
        self.model_name = model_name
        self.hp = params
        self.layer_list = []
        self.output_v = None
        self.layer_count = 0
        self.accuracy = 0
        self.loss_func = None
        self.subfolder = os.getcwd() + "/" + self.__create_subfolder()

    def add_layer(self, layer, name):
        layer.initialize(self.subfolder, name)
        self.layer_list.append(layer)
        self.layer_count += 1

    def add_layers(self, layers: list, name):
        for layer in layers:
            # layer.initialize(self.subfolder, iteration_count)
            self.layer_list.append(layer)
            self.layer_count += 1

    def __forward(self, input_v, train=True):
        output = None
        for i in range(self.layer_count):
            try:
                output = self.layer_list[i].forward(input_v, train)
            except:
                print(i)
            input_v = output

        self.output_v = output
        return self.output_v

    def inference(self, input_v):
        output = self.__forward(input_v, train=False)
        return output

    def __backward(self, Y):
        delta_in = self.output_v - Y
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            delta_out = layer.backward(delta_in, i)
            delta_in = delta_out

    def __update(self):
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            layer.update()

    def train(self, dataReader, checkpoint=0.1, need_test=True, file_name=""):
        t0 = time.time()
        self.loss_func = LossFunction(self.hp.net_type)
        if self.hp.regular_name == RegularMethod.EarlyStop:
            self.loss_trace = TrainingHistory(True, self.hp.regular_value)
        else:
            self.loss_trace = TrainingHistory()

        if self.hp.batch_size == -1 or self.hp.batch_size > dataReader.num_train:
            self.hp.batch_size = dataReader.num_train

        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = int(math.ceil(max_iteration * checkpoint))
        need_stop = False
        valid_count = 0
        for epoch in range(self.hp.max_epoch):
            dataReader.training_Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # for optimizers which need pre-update weights
                # if self.hp.optimizer_name == OptimizerName.Nag:
                #     self.__pre_update()
                # get z from x,y

                time1 = time.time()
                self.__forward(batch_x, train=True)
                time2 = time.time()
                # calculate gradient of w and b
                self.__backward(batch_y)
                time3 = time.time()
                # final update w,b
                self.__update()
                time4 = time.time()
                print(f"iteration {iteration} , forward time: {time2 - time1}, "
                      f"backward time: {time3 - time2}, update time: {time4 - time3},total time: {time4 - time1}")

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    if batch_x.shape[0] < self.hp.batch_size:
                        batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration-1)
                    self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
                    self.SaveLossHistory(valid_count,name=file_name)
                    valid_count = valid_count + 1
                    if need_stop:
                        break

            self.save_parameters()
            if need_stop:
                break
            # end if
        # end for

        t1 = time.time()
        print("time used:", t1 - t0)

        if need_test:
            print("testing...")
            self.accuracy = self.Test(dataReader)
            print(self.accuracy)

    def distributed_train(self, batch_x, batch_y):
        time1 = time.time()
        self.__forward(batch_x, train=True)
        time2 = time.time()
        self.__backward(batch_y)
        time3 = time.time()
        # self.__update()
        # time4 = time.time()
        print(f"forward time: {time2 - time1}, "
              f"backward time: {time3 - time2}, total time: {time3 - time1}")

        grad = self.distributed_save_gradient()
        return grad

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" % (epoch, total_iteration))

        # l1/l2 cost
        regular_cost = self.__get_regular_cost(self.hp.regular_name)

        # calculate train loss
        self.__forward(train_x, train=False)
        loss_train, accuracy_train = self.loss_func.CheckLoss(self.output_v, train_y)
        loss_train = loss_train + regular_cost / train_x.shape[0]
        print("loss_train=%.6f, accuracy_train=%f" % (loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.__forward(vld_x, train=False)
        loss_vld, accuracy_vld = self.loss_func.CheckLoss(self.output_v, vld_y)
        loss_vld = loss_vld + regular_cost / vld_x.shape[0]
        print("loss_valid=%.6f, accuracy_valid=%f" % (loss_vld, accuracy_vld))

        # end if
        need_stop = self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld,
                                        self.hp.stopper)
        if self.hp.stopper is not None:
            if self.hp.stopper.stop_condition == StopCondition.StopLoss and loss_vld <= self.hp.stopper.stop_value:
                need_stop = True
        return need_stop

    def __get_regular_cost(self, regularName):
        if regularName != RegularMethod.L1 and regularName != RegularMethod.L2:
            return 0

        regular_cost = 0
        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layer_list[i]
            if isinstance(layer,MiniFramework.FCLayer) or isinstance(layer,MiniFramework.ConLayer):
                if regularName == RegularMethod.L1:
                    regular_cost += np.sum(np.abs(layer.WB.W))
                elif regularName == RegularMethod.L2:
                    regular_cost += np.sum(np.square(layer.WB.W))
            # end if
        # end for
        return regular_cost * self.hp.regular_value

    def Test(self, dataReader):
        x, y = dataReader.GetTestSet()
        self.__forward(x, train=False)
        _, correct = self.loss_func.CheckLoss(self.output_v, y)
        return correct

    # save weights value when got low loss than before
    def save_parameters(self):
        print("save parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.save_parameters()

    def distributed_save_parameters(self):
        print("save parameters")
        params = {}
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            try:
                params.update(layer.distributed_save_parameters())
            except TypeError:
                pass
        return params

    def distributed_save_gradient(self):
        print("save gradient")
        grad = {}
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            try:
                grad.update(layer.distributed_save_gradient())
            except TypeError:
                pass
        return grad

    # load weights for the most low loss moment
    def load_parameters(self):
        print("load parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.load_parameters()

    def distributed_load_parameters(self, param):
        print("load parameters")
        name = list(param.keys())[0]
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.distributed_load_parameters(param[name])

    def distributed_load_gradient(self,grad):
        print("load gradient")
        name = list(grad.keys())[0]
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.distributed_load_gradient(grad[name])

    def distributed_add_gradient(self, grad):
        print("add parameters")
        name = list(grad.keys())[0]
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.distributed_add_gradient(grad[name])

    def distributed_average_gradient(self, num):
        print("average parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.distributed_average_gradient(num)

    def ShowLossHistory(self, xcoor, xmin=None, xmax=None, ymin=None, ymax=None):
        title = str.format("{0},accuracy={1:.4f}", self.hp.toString(), self.accuracy)
        self.loss_trace.ShowLossHistory(title, xcoor, xmin, xmax, ymin, ymax)

    def SaveLossHistory(self,valid_count,name="loss_data"):
        # path = self.subfolder

        name_attribute = ['epoch', 'iteration', 'training_loss', 'training_accuracy', 'validating_loss',
                          'validating_accuracy']
        csvFile = open(name, "a+", newline='')
        try:
            writer = csv.writer(csvFile)
            if valid_count == 0:
                writer.writerow(name_attribute)

            writer.writerow(
                    [self.loss_trace.epoch_seq[valid_count], self.loss_trace.iteration_seq[valid_count], self.loss_trace.training_loss[valid_count],
                     self.loss_trace.training_accuracy[valid_count], self.loss_trace.val_loss[valid_count],
                     self.loss_trace.val_accuracy[valid_count]])
        finally:
            csvFile.close()

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("/")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path
