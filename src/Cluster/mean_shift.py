__author__ = 'dengzhihong'

from src.Cluster.base import *
import numpy as np
from src.Methods.math_methods import *
from src.Methods.process_data import *
from src.Methods.draw_diagram import *

class MeanShift(ClusterBase):
    @staticmethod
    def getNextShift(X, Point, Mat_h):
        N = X.shape[0]
        D = X.shape[1]
        A_Part = np.zeros((1, D))
        B_Part = np.zeros((1, D))
        for i in range(N):
            A_Part += X[i] * multivariateGaussian(X[i], Point, Mat_h)
            B_Part += multivariateGaussian(X[i], Point, Mat_h)
        NextPoint = A_Part/B_Part
        return NextPoint

    @staticmethod
    def getPeak(X, i, h):
        Threshold = 0.0001
        N = X.shape[0]
        D = X.shape[1]
        Old_Point = np.zeros((1, D))
        Point = np.zeros((1, D))
        Mat_h = (h**2) * np.eye(D)
        Old_Point = X[i]
        print 'Initial Point: ', Old_Point
        while(True):
            Point = MeanShift.getNextShift(X, Old_Point, Mat_h)
            print 'NextPoint: ', Point
            #break
            if(ClusterBase.isVectorConverge(Point, Old_Point, Threshold)):
                print 'Converge!'
                break
            Old_Point = Point
        return np.array(Point).reshape(-1)

    '''
    @staticmethod
    def isConverge(Point, Old_Point, Threshold):
        #return (abs(Point - Old_Point) < Threshold).any()
        return (abs(Point - Old_Point) < Threshold).all()
    '''

    @staticmethod
    def find(PeakSet, PeakPoint, Threshold):
        N = len(PeakSet)
        for i in range(N):
            if(ClusterBase.isVectorConverge(PeakSet[i], PeakPoint, Threshold)):
                return i
        return -1

    @staticmethod
    def clusteringWithPeak(PeakPoint, Threshold):
        N = PeakPoint.shape[0]
        D = PeakPoint.shape[1]
        Z = []
        PeakSet = []
        count = -1
        ClassIndex = []
        Threshold = 0.1
        #print 'PeakPoint: '
        #print PeakPoint
        for i in range(N):
            #print 'Count = ', count
            #print 'The ', i ,'th point'
            index = MeanShift.find(PeakSet, PeakPoint[i], Threshold)
            # Not Exist
            if(index == -1):
                #print 'Not Exist'
                PeakSet.append(PeakPoint[i])
                count += 1
                ClassIndex.append(count)
                Z.append(count)
                #print 'Allocate to ', count
            # Exist
            else:
                #print 'Exist, belong to ', ClassIndex[index]
                Z.append(ClassIndex[index])
        #print 'PeakSet:'
        #print PeakSet
        #print 'Z:'
        #print Z
        return Z

    @staticmethod
    def runMeanShift(X, h):
        N = X.shape[0]
        D = X.shape[1]
        PeakPoint = np.zeros((N, D))

        for i in range(N):
            print 'Point ', i
            PeakPoint[i] = MeanShift.getPeak(X, i, h)
            print '-------------------------------------'
        #print PeakPoint

        Threshold = 0.3
        Z = MeanShift.clusteringWithPeak(PeakPoint, Threshold)
        return Z

    @staticmethod
    def showMeanShiftInCluster(data, Z, Title, BandWidth):
        X = np.array(data)
        N = X.shape[0]
        ClusterX = []
        ClusterY = []
        ClassX = []
        ClassY = []
        ClassIndex = -1
        count = 0
        while(True):
            if(count == N):
                break
            for i in range(N):
                if(Z[i] != -1):
                    ClassIndex = Z[i]
                    break
            for i in range(N):
                if(Z[i] == ClassIndex):
                    ClassX.append(X[i][0])
                    ClassY.append(X[i][1])
                    Z[i] = -1
                    count += 1
            ClusterX.append(ClassX)
            ClusterY.append(ClassY)
            ClassX = []
            ClassY = []
        K = len(ClusterX)
        ColorPattern = ['r.', 'g.', 'b.', 'y.', 'k.', 'm.', 'c.', 'wo', 'ro','go','bo','yo','ko','mo','co', 'w-']
        ax = plt.figure().add_subplot(111)
        ax.set_title(Title + ' BandWidth = ' + str(BandWidth) , fontsize = 18)
        plt.axis([-20, 20, -20, 20])
        ax.xaxis.grid(True, which='major')
        ax.yaxis.grid(True, which='major')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        for k in range(K):
            plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
        plt.legend()
        plt.show()

    @staticmethod
    def testWithMeanShift(data, title, Bandwidth):
        X = RawData2XYArray(data)
        Z = MeanShift.runMeanShift(X, Bandwidth)
        MeanShift.showMeanShiftInCluster(X, Z, title, Bandwidth)