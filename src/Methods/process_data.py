__author__ = 'dengzhihong'

import numpy as np

def str2FloatList(stringlist):
    floatlist = []
    for i in range(0, len(stringlist)):
        floatlist.append(float(stringlist[i]))
    return floatlist

def str2IntList(stringlist):
    intlist = []
    for i in range(0, len(stringlist)):
        intlist.append(int(float(stringlist[i])))
    return intlist

def getCoordFromList(data):
    R = 2
    L = len(data)
    data = np.array(str2FloatList(data)).reshape(R,L/R)
    X = []
    Y = []
    for i in range(data.shape[1]):
        X.append(data[0][i])
        Y.append(data[1][i])
    return X,Y

def preprocessRawData(data):
    dataMat = np.array(str2FloatList(data)).reshape(2, len(data)/2)
    return np.mat(dataMat).transpose()