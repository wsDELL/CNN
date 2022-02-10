import matplotlib.pyplot as plt
import pickle

from MiniFramework.Enums import *


class TrainingHistory(object):
    def __init__(self, need_earlyStop=False, patience=5):
        self.training_loss = []
        self.training_accuracy = []
        self.iteration_seq = []
        self.epoch_seq = []
        self.val_loss = []
        self.val_accuracy = []
        self.counter = 0

        self.early_stop = need_earlyStop
        self.patience = patience
        self.patience_counter = 0
        self.last_vld_loss = float("inf")

    def Add(self, epoch, total_iteration, training_loss, training_accuracy, val_loss, val_accuracy, stopper):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.training_loss.append(training_loss)
        self.training_accuracy.append(training_accuracy)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_accuracy is not None:
            self.val_accuracy.append(val_accuracy)

        if stopper is not None:
            if stopper.stop_condition == StopCondition.StopDiff:
                if len(self.val_loss) > 1:
                    if abs(self.val_loss[-1] - self.val_loss[-2]) < stopper.stop_value:
                        self.counter = self.counter + 1
                        if self.counter > 3:
                            return True
                    else:
                        self.counter = 0

        if self.early_stop:
            if val_loss < self.last_vld_loss:
                self.patience_counter = 0
                self.last_vld_loss = val_loss
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    return True
        return False

    def ShowLossHistory(self, title, xcoord, xmin=None, xmax=None, ymin=None, ymax=None):
        fig = plt.figure(figsize=(12, 5))

        axes = plt.subplot(1, 2, 1)
        if xcoord == XCoordinate.Iteration:
            p2, = axes.plot(self.iteration_seq, self.training_loss)
            p1, = axes.plot(self.iteration_seq, self.val_loss)
            axes.set_xlabel("iteration")
        elif xcoord == XCoordinate.Epoch:
            p2, = axes.plot(self.epoch_seq, self.training_loss)
            p1, = axes.plot(self.epoch_seq, self.val_loss)
            axes.set_xlabel("epoch")
        # end if
        axes.legend([p1, p2], ["validation", "train"])
        axes.set_title("Loss")
        axes.set_ylabel("loss")
        if xmin != None or xmax != None or ymin != None or ymax != None:
            axes.axis([xmin, xmax, ymin, ymax])

        axes = plt.subplot(1, 2, 2)
        if xcoord == XCoordinate.Iteration:
            p2, = axes.plot(self.iteration_seq, self.training_accuracy)
            p1, = axes.plot(self.iteration_seq, self.val_accuracy)
            axes.set_xlabel("iteration")
        elif xcoord == XCoordinate.Epoch:
            p2, = axes.plot(self.epoch_seq, self.training_accuracy)
            p1, = axes.plot(self.epoch_seq, self.val_accuracy)
            axes.set_xlabel("epoch")
        # end if
        axes.legend([p1, p2], ["validation", "train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")

        plt.suptitle(title)
        plt.show()
        return title

    def GetEpochNumber(self):
        return self.epoch_seq[-1]

    def GetLatestAverageLoss(self, count=10):
        total = len(self.val_loss)
        if count >= total:
            count = total
        tmp = self.val_loss[total - count:total]
        return sum(tmp) / count

    def Dump(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self, f)

    def Load(self,file_name):
        f = open(file_name, 'rb')
        lh = pickle.load(f)
        return lh