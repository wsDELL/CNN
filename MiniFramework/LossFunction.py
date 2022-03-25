import numpy as np

from MiniFramework.Enums import *
from MiniFramework.util import *


class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type

    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        loss = None
        acc = None
        if self.net_type == NetType.Fitting:
            loss, acc = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss, acc = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss, acc = self.CE3(A, Y, m)

        if loss.ndim == 1:
            return loss[0], acc[0]
        return loss, acc

    def MSE(self, A, Y, count):
        # loss
        p1 = A - Y
        LOSS = np.multiply(p1, p1)
        loss = np.sum(LOSS) / count / 2
        # accuracy
        var = np.var(Y)
        mse = np.sum((A - Y) * (A - Y)) / count
        r2 = 1 - mse / var

        return loss, r2

    # for binary classifier
    def CE2(self, A, Y, count):
        # loss
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)
        p4 = np.multiply(p1, p2)
        p5 = np.multiply(Y, p3)
        LOSS = np.sum(-(p4 + p5))
        loss = LOSS / count
        # accuracy
        b = np.round(A)
        r = (b == Y)
        correct = np.sum(r) / count

        return loss, correct

    # for multiple classifier
    def CE3(self, A, Y, count):
        # loss
        p1 = np.log(A + 1e-7)
        p2 = np.multiply(Y, p1)
        LOSS = np.sum(-p2)
        loss = LOSS / count
        # acc
        ra = np.argmax(A, axis=1)
        ry = np.argmax(Y, axis=1)
        r = (ra == ry)
        correct = np.sum(r) / count

        return loss, correct

    def Zero_OneLoss(self, A, Y, count):
        for i in count:
            LOSS = 0
            if A[i] != Y[i]:
                LOSS += 1
        loss = LOSS / count
        acc = 1 - loss
        return loss, acc


def CrossEntropyLoss(A, Y):
    p1 = np.log(A + 1e-7)
    p2 = np.multiply(Y, p1)
    loss = -p2
    return loss


def BinaryCrossEntropyLoss(A, Y):
    p1 = 1 - Y
    p2 = np.log(1 - A)
    p3 = np.log(A)
    p4 = np.multiply(p1, p2)
    p5 = np.multiply(Y, p3)
    loss = -(p4 + p5)
    return loss

def MSE(A, Y):
    # loss
    p1 = A - Y
    loss = np.multiply(p1, p1)
    return loss
