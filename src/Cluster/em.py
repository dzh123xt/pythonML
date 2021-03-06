__author__ = 'dengzhihong'

import numpy as np
from src.Cluster.base import *
from src.Methods.math_methods import *
from src.Methods.draw_diagram import *
from src.Methods.process_data import *
import random

class EM(ClusterBase):

    @staticmethod
    def E_Step(X, Mean, Cov, Pai):
        K = Mean.shape[0]
        N = X.shape[0]
        Z = np.zeros((N, K))
        Temp = 0
        Sum = []
        for i in range(N):
            for k in range(K):
                '''
                print k
                print 'X[i]: ', X[i]
                print 'Mean[k]: ', Mean[k]
                print 'Cov[k]: ', Cov[k]
                print 'Pai[k]: ', Pai[k]
                '''
                #a = Pai[k] * multivariateGaussian( transpose(X[i]), transpose(Mean[k]), Cov[k])
                a = Pai[k] * multivariateGaussian( X[i], Mean[k], Cov[k])
                Temp += a
            Sum.append(Temp)
            Temp = 0
        for i in range(N):
            for j in range(K):
                #b = Pai[j] * multivariateGaussian( transpose(X[i]), transpose(Mean[j]), Cov[j])
                b = Pai[j] * multivariateGaussian( X[i], Mean[j], Cov[j])
                #print 'Z[i][j]: ', Z[i][j]
                #print 'Sum[i]: ', Sum[i]
                #print 'b: ', b
                Z[i][j] = b / Sum[i]

        for i in range(N):
            if(EM.sumLine(Z[i]) - 1.0 > 0.00001):
                print 'False'
                print EM.sumLine(Z[i])
        return Z

    @staticmethod
    def sumLine(array):
        N = 4
        sum = 0
        for i in range(N):
            sum += float(array[i])
        return sum

    @staticmethod
    def M_Step(X, Z):
        N = Z.shape[0]
        K = Z.shape[1]
        D = X.shape[1]
        Mean = np.zeros((K, 1, D))
        Cov = np.zeros((K, D, D))
        TempCov = np.zeros((D, D))
        Pai = []
        Meanj = np.zeros((1, D))
        Nj = []
        TempNj = 0
        Covj = np.zeros((D,D))
        # Get Nj
        for j in range(K):
            for i in range(N):
                TempNj += Z[i][j]
            Nj.append(TempNj)
            TempNj = 0
        #print 'Nj:'
        #print Nj
        # Get Paij
        for j in range(K):
            Pai.append(Nj[j]/N)
        #print 'Pai:'
        #print Pai
        # Get Meanj
        for j in range(K):
            for i in range(N):
                Meanj += Z[i][j] * X[i]
            Meanj /= Nj[j]
            Mean[j] = Meanj
            Meanj = np.zeros((1, D))
        #print 'Mean: '
        # Get Cov
        for j in range(K):
            for i in range(N):
                #print '---'
                #print TempCov
                TempCov += Z[i][j] * np.dot(np.transpose(X[i] - Mean[j]), X[i] - Mean[j])
            Cov[j] = TempCov/Nj[j]
            TempCov = np.zeros((D, D))
        #print Cov
        return Mean, Cov, Pai

    @staticmethod
    def getQ(X, Z, Mean, Cov, Pai):
        N = Z.shape[0]
        K = Z.shape[1]
        Q = 0

        for i in range(N):
            for j in range(K):
                #Q += Z[i][j] * math.log(multivariateGaussian(X[i], Mean[j], Cov[j]) * Pai[j])
                Q += Z[i][j] * multivariateGaussian(X[i], Mean[j], Cov[j]) * Pai[j]
        return Q

    @staticmethod
    def genRandCov(data, K):
        D = data.shape[1]
        o_cov =  np.cov(data.transpose())
        Cov = np.zeros((K, D, D))
        for k in range(K):
           Cov[k] = o_cov + np.random.rand(2,2)
        return Cov

    @staticmethod
    def genPai(K):
        Pai = []
        P = 0
        for k in range(K):
            Pai.append(1.0/K)
        for i in range(K):
            p = random.uniform(0, 1.0/K - 0.05)
            Pai[i] -= p
            Pai[K - 1 - i] += p
        return np.array(Pai)

    @staticmethod
    def runEM(data, K, iteration):
        D = data.shape[1]
        N = data.shape[0]
        Z = np.zeros((N,1))
        # Generate initial Mean which contains 4 matrix of 2x2
        Mean = ClusterBase.genRandMean(data, K)
        # Generate initial Cov
        Cov = EM.genRandCov(data, K)
        # Generate pai
        print 'Mean: \n', Mean
        print 'Cov: \n', Cov
        Pai = EM.genPai(K)
        print 'Pai: \n', Pai
        count = 0
        Q_old = -9999999
        Q_new = 0
        while(True):
            print count
            print '+++++++++++++++++++++++++++++++++++++++++'
            #displayData(Z, Mean, Cov, Pai, 'Old Parameter')
            print 'E-Step'
            Z = EM.E_Step(data, Mean, Cov, Pai)
            #print Z
            print 'M-Step'
            Mean, Cov, Pai = EM.M_Step(data, Z)
            #displayData(Z, Mean, Cov, Pai, 'New Parameter')
            Q_new =  EM.getQ(data, Z, Mean, Cov, Pai)
            print 'Q = ', Q_new
            '''
            if(Q_new > Q_old  and (Q_new - Q_old < 0.000000000001)):
                print 'Converge!!'
                break
            '''
            Q_old = Q_new
            if(count == iteration):
                print 'Iteration Reached'
                break
            count += 1

        Z_Normal = np.zeros((N,1))
        for i in range(N):
            index = EM.findMax(Z[i])
            Z_Normal[i] = index
        return Z_Normal.astype(int).reshape(-1).tolist()

    @staticmethod
    def findMax(list):
        MaxIndex = -1
        Max = -999999
        #print list
        for i in range(len(list)):
            if(list[i] > Max):
                Max = list[i]
                MaxIndex = i
        return MaxIndex

    @staticmethod
    def testWithEM(data, title, K):
        X = RawData2XYArray(data)
        Z = EM.runEM(X, K, 100)
        showDiagramInCluster(X, Z, "EM_" + title)