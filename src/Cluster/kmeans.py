__author__ = 'dengzhihong'

from src.Cluster.base import *
import numpy as np
from src.Methods.math_methods import *
from src.Methods.process_data import *
from src.Methods.draw_diagram import *

class KMeans(ClusterBase):
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
                norm = getSquareNorm(data[i], Mean[k])
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
        Mean = np.mat(np.zeros((K,D)))
        for j in range(K):
            DataInClusterJ = data[np.nonzero(label[:,0] == j)]
            Mean[j,:] = np.mean(DataInClusterJ, axis=0)
        return Mean

    @staticmethod
    def runKmeans(data, K):
        D = data.shape[1]
        N = data.shape[0]
        Z = np.zeros((N, 1))
        Mean = ClusterBase.genRandMean(data, K)
        print 'Initial Mean: '
        print Mean
        count = 0
        Mean_Old = np.mat(np.zeros((K, D)))
        while(True):
            print '++++++++++++++++++++++++++++++++'
            print count
            Z = KMeans.clusterAssignment(data, Mean)
            Mean = KMeans.estimateCenter(data, Z, K)
            print Mean
            count +=1
            if(KMeans.isVectorConverge(Mean, Mean_Old, 0.0001)):
                print 'Converge'
                break
            Mean_Old = Mean
        return Z.reshape(-1).astype(int).tolist()

    @staticmethod
    def testWithKmeans(data, title, K):
        X = RawData2XYArray(data)
        Z = KMeans.runKmeans(X, K)
        showDiagramInCluster(X, Z, "Kmeans_" + title)