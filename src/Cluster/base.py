__author__ = 'dengzhihong'

import numpy as np
import matplotlib.pyplot as plt
import random
import math

class CommonMethod(object):
    @staticmethod
    def str2FloatList(stringlist):
        floatlist = []
        for i in range(0, len(stringlist)):
            floatlist.append(float(stringlist[i]))
        return floatlist

    @staticmethod
    def str2IntList(stringlist):
        intlist = []
        for i in range(0, len(stringlist)):
            intlist.append(int(float(stringlist[i])))
        return intlist

    @staticmethod
    def getCoordFromList(data):
        R = 2
        L = len(data)
        data = np.array(CommonMethod.str2FloatList(data)).reshape(R,L/R)
        X = []
        Y = []
        for i in range(data.shape[1]):
            X.append(data[0][i])
            Y.append(data[1][i])
        return X,Y

    @staticmethod
    def preprocessRawData(data):
        dataMat = np.array(CommonMethod.str2FloatList(data)).reshape(2, len(data)/2)
        return np.mat(dataMat).transpose()


    @staticmethod
    def multivariateGaussian(vector_x, vector_mean, mat_covariance):
        k = vector_x.shape[0]
        first = 1 / math.sqrt((2 * math.pi * np.linalg.det(mat_covariance))**k)
        sub = vector_x - vector_mean
        #norm = -0.5 * dot( dot( transpose(sub), inv(mat_covariance)),  sub)
        norm = -0.5 * np.dot( np.dot( sub, np.linalg.inv(mat_covariance)),  np.transpose(sub))
        result = first * pow(math.e, float(norm))
        return result

    @staticmethod
    def sumLine(array):
        N = 4
        sum = 0
        for i in range(N):
            sum += float(array[i])
        return sum

    @staticmethod
    def findMaxFromList(list):
        MaxIndex = -1
        Max = -999999
        #print list
        for i in range(len(list)):
            if(list[i] > Max):
                Max = list[i]
                MaxIndex = i
        return MaxIndex

    @staticmethod
    def getSquareNorm(X, Mean):
        sub = np.transpose(X) - np.transpose(Mean)
        result = np.dot( np.transpose(sub), sub)
        return float(result)

    #------------------------------------------------------------------------------------------------------
    # For drawing
    @staticmethod
    def drawDiagram(x, y, title="", directory="", MethodName=""):
        ax = plt.figure().add_subplot(111)
        ax.set_title(title + '   Algorithm: '+ MethodName, fontsize = 18)
        plt.axis([-20,20,-20,20])
        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.plot(x, y,'b.')
        plt.savefig('./Diagram/' + directory + '/' + MethodName + '.jpg', dpi=200)

    @staticmethod
    def showDiagram(x, y, title="", MethodName=""):
        ax = plt.figure().add_subplot(111)
        ax.set_title(title + '   Algorithm: '+ MethodName, fontsize = 18)
        plt.axis([-20,20,-20,20])
        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.plot(x, y,'b.')
        plt.show()

    @staticmethod
    def showDiagramInCluster(data, Z, title=""):
        N = data.shape[0]
        X = np.array(data)
        ColorPattern = ['r.', 'g.', 'b.', 'y.', 'k.', 'm.', 'c.']

        ClusterX = [[],[],[],[]]
        ClusterY = [[],[],[],[]]
        for i in range(N):
            ClusterX[Z[i]].append(X[i][0])
            ClusterY[Z[i]].append(X[i][1])

        K = len(ClusterX)
        ax = plt.figure().add_subplot(111)
        ax.set_title('Real clustering of ' + title  , fontsize = 18)
        plt.axis([-20,20,-20,20])
        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        for k in range(K):
            #index = Z[i] - 1
            plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
        plt.show()

    @staticmethod
    def drawDiagramInCluster(X, Z, Title="", MethodName="", Directory=""):
        D = X.shape[2]
        N = X.shape[0]
        K = Z.shape[1]
        Class = []
        ColorPattern = ['r.', 'g.', 'b.', 'y.', 'k.', 'm.', 'c.', 'ro']
        for i in range(N):
            for k in range(K):
                if(Z[i][k] == 1.0):
                    Class.append(k)
                    break
        ClusterX = [[],[],[],[]]
        ClusterY = [[],[],[],[]]
        for i in range(N):
            ClusterX[Class[i]].append(X[i][0][0])
            ClusterY[Class[i]].append(X[i][0][1])

        ax = plt.figure().add_subplot(111)
        ax.set_title(Title + '   Algorithm: '+ MethodName, fontsize = 18)
        plt.axis([-20,20,-20,20])
        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        for k in range(K):
            plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
        plt.legend()
        plt.savefig('./Diagram/' + Directory + '/' + MethodName + Title +'.jpg', dpi=200)

    @staticmethod
    def drawLabeledDiagram(dataA_X, dataA_Y, title):
        x, y = CommonMethod.getCoordFromList(dataA_X)
        Z = CommonMethod.str2IntList(dataA_Y)
        CommonMethod.showDiagramInCluster(x, y, Z, title)
