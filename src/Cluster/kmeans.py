__author__ = 'dengzhihong'

from src.Cluster.base import *
import numpy as np

class KMeans(CommonMethod):
    @staticmethod
    def clusterAssignment(data, Mean):
        D = data.shape[1]
        K = Mean.shape[0]
        N = data.shape[0]
        min = np.inf
        Z = np.zeros((N, 1))
        index = -1
        for i in range(N):
            for k in range(K):
                norm = KMeans.getSquareNorm(data[i], Mean[k])
                #print norm
                if(norm < min):
                    index = k
                    min = norm
            min = np.inf
            Z[i] = index
            index = -1
        return Z

    @staticmethod
    def estimateCenter(data, label, K):
        D = data.shape[1]
        Mean = np.zeros((K,D))
        for j in range(K):
            DataInClusterJ = data[np.nonzero(label[:,0] == j)]
            Mean[j,:] = np.mean(DataInClusterJ, axis=0)
        return Mean

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
    def isVectorConverge(v1, v2):
        threshold = 0.0001
        if((abs(v1 - v2) > threshold).all() == False):
            return True
        else:
            return False

    @staticmethod
    def runKmeans(data, K):
        D = data.shape[1]
        N = data.shape[0]
        Z = np.zeros((N, 1))
        Mean = KMeans.genRandMean(data, K)
        print 'Initial Mean: '
        print Mean
        count = 0
        Z_Old = np.zeros((N,1))
        while(True):
            print '++++++++++++++++++++++++++++++++'
            print count

            Z = KMeans.clusterAssignment(data, Mean)
            Mean = KMeans.estimateCenter(data, Z, K)

            print Mean
            count +=1
            if(KMeans.isVectorConverge(Z, Z_Old)):
                print 'Converge'
                print Z.transpose()
                break
        return Z.reshape(-1).astype(int).tolist()

    @staticmethod
    def showDiagramInCluster(X, Z, title="Kmeans"):
        CommonMethod.showDiagramInCluster(X, Z, title)

    @staticmethod
    def testWithKmeans(data, title, K):
        X = CommonMethod.preprocessRawData(data)
        Z = KMeans.runKmeans(X, K)
        KMeans.showDiagramInCluster(X, Z)