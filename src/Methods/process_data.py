__author__ = 'dengzhihong'

import numpy as np

def RawLabel2IntList(label):
    return np.array(map(int, map(float, label[:]))) - 1

def RawData2FloatXYList(data):
    R = 2
    L = len(data)
    data = np.array(map(float, data[:])).reshape(R,L/R)
    X = []
    Y = []
    for i in range(data.shape[1]):
        X.append(data[0][i])
        Y.append(data[1][i])
    return X,Y

def RawData2XYArray(data):
    dataMat = np.array(map(float, data[:])).reshape(2, len(data)/2)
    return np.mat(dataMat).transpose()