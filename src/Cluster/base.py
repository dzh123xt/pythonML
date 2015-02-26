__author__ = 'dengzhihong'

import numpy as np

class ClusterBase(object):
    @staticmethod
    def genRandMean(data, K):
        D = data.shape[1]
        RandMean = np.mat(np.zeros((K, D)))
        for i in range(D):
            LowValue = np.min(data[:,i])
            RangeValue = float(np.max(data[:,i]) - LowValue)
            RandMean[:,i] = LowValue + RangeValue * np.random.rand(K, 1)
        return RandMean

    @staticmethod
    def isVectorConverge(v1, v2, threshold):
        if((abs(np.array(v1) - np.array(v2)) < threshold).all() == True):
            return True
        else:
            return False