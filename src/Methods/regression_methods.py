__author__ = 'dengzhihong'

from src.Regression.ls import *
from src.Regression.rls import *
from src.Regression.rr import *
from src.Regression.br import *
from src.Regression.lasso import *
import numpy as np
from src.Methods.draw_diagram import *
import random

def testWithRegression(sampx, sampy, polyx, polyy, K, MethodList):
     y = RegressionBase.toFloatList(polyy)
     for method in MethodList:
        SquareMean = 0
        prediction = Regression(sampx, sampy, polyx, polyy, K, method)
        for i in range(len(prediction)):
            SquareMean += (prediction[i] - y[i])**2
        SquareMean /= len(prediction)
        print method
        print 'Square Mean Error = ', SquareMean

def testWithReduction(sampx, sampy, polyx, polyy, K, MethodList):
    ResultList = []
    TestTimes = 1
    ReductionRateList = [0, 15, 30, 45, 60, 75, 90]
    for rate in ReductionRateList:
        for method in MethodList:
            ResultList.append( regressionWithReduction(sampx, sampy, polyx, polyy, K, TestTimes, rate, method) )
    x = np.array(ResultList).reshape(len(ReductionRateList), len(MethodList)).tolist()

    for row in x:
        for col in row:
            print str(col) + '\t',
        print

    for j in range(len(MethodList)):
        MeanError = []
        for i in range(len(ReductionRateList)):
            MeanError.append(x[i][j])
        showMeanErrorDiagram(ReductionRateList, MeanError, 'Average MeanError, Method = ', MethodList[j])

def testWithLargeValue(sampx, sampy, polyx, polyy, K, MethodList):
    N = len(sampx)
    Num = 5
    randlist = []
    for i in range(Num):
        r = random.randint(0, N-1)
        while(randlist.count(r) != 0):
            r = random.randint(0,N-1)
        randlist.append(r)
    for i in randlist:
        sampy[i] = str(float(sampy[i]) + 200)
    for method in MethodList:
        Regression(sampx, sampy, polyx, polyy, K, method)

def testWithHigherK(sampx, sampy, polyx, polyy, K, MethodList):
    for method in MethodList:
        Regression(sampx, sampy, polyx, polyy, K, method)

def Regression(sampx, sampy, polyx, polyy, K, method):
    sigma = 0
    if(method == 'LS'):
        theta = LS.run(sampx, sampy, K)
    elif(method == 'RLS'):
        theta = RLS.run(sampx, sampy, K)
    elif(method == 'LASSO'):
        theta = LASSO.run(sampx, sampy, K)
    elif(method == 'RR'):
        theta = RR.run(sampx, sampy, K)
    elif(method == 'BR'):
        theta, sigma = BR.run(sampx, sampy, K)
    else:
        print 'No method'
        return 0
    prediction = RegressionBase.getPredictionValueList(polyx, theta, K)
    if(method == 'BR'):
        showRegressionDiagramBR(sampx, sampy, polyx, polyy, prediction, theta, sigma, K, method)
    else:
        showRegressionDiagramExceptBR(sampx, sampy, polyx, polyy, prediction,  method)
    return prediction

def regressionWithReduction(sampx, sampy, polyx, polyy, K, TestTimes, ReducitionRate, method):
    print str(K) + 'th ' + 'TestTimes = ' + str(TestTimes) + ' ReductionRate = ' + str(ReducitionRate) + '% Method = ' + method
    MeanErrorScalar = 0.0
    PredictionSum = np.zeros(len(polyx))
    for i in range(TestTimes):
        x = sampx[:]
        y = sampy[:]
        Theta, x, y = getThetaWithReduction(x, y, K, ReducitionRate, method)
        Prediction = BR.getPredictionValueList(polyx, Theta, K)
        #PredictionSum += Prediction
        if(i == 0):
            showPredictionDiagramWithReduction(x, y, polyx, polyy, Prediction, K, ReducitionRate, method)
        MeanErrorVector = np.array(RegressionBase.strlistToFloatvector(Prediction) - RegressionBase.strlistToFloatvector(polyy))
        MeanErrorScalar += np.dot(MeanErrorVector.T, MeanErrorVector)

    #PredictionSum /= TestTimes
    MeanErrorScalar /= (TestTimes * len(polyy))
    MeanErrorScalar = float(MeanErrorScalar)

    return MeanErrorScalar

def getThetaWithReduction(sampx, sampy, K, ReductionRate, method):
    N = len(sampx)
    reduction = int(N * ReductionRate * 0.01)
    rest = N - reduction
    randlist = []
    new_x = []
    new_y = []
    for i in range(rest):
        r = random.randint(0, N-1)
        while(randlist.count(r) != 0):
            r = random.randint(0,N-1)
        randlist.append(r)

    for i in randlist:
        new_x.append(sampx[i])
        new_y.append(sampy[i])

    if(method == 'LS'):
        theta = LS.run(new_x, new_y, K)
    elif(method == 'RLS'):
        theta = RLS.run(new_x, new_y, K)
    elif(method == 'LASSO'):
        theta = LASSO.run(new_x, new_y, K)
    elif(method == 'RR'):
        theta = RR.run(new_x, new_y, K)
    elif(method == 'BR'):
        theta, variance = BR.run(new_x, new_y, K)
    else:
        print 'No method'
        return 0
    return theta, new_x, new_y
