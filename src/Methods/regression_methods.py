__author__ = 'dengzhihong'

from src.Regression.ls import *
from src.Regression.rls import *
from src.Regression.rr import *
from src.Regression.br import *
from src.Regression.lasso import *
import numpy as np

def testWithRegression(sampx, sampy, polyx, polyy, K, MethodList, choice):
     y = map(float, polyy)
     for method in MethodList:
        SquareMean = 0
        prediction = Regression(sampx, sampy, polyx, polyy, K, method, 'b', choice)

        for i in range(len(prediction)):
            SquareMean += (prediction[i] - y[i])**2
        SquareMean /= len(prediction)
        print method
        print 'Square Mean Error = ', SquareMean

def testWithReduction(sampx, sampy, polyx, polyy, K, MethodList, choice):
    ResultList = []
    TestTimes = 1
    ReductionRateList = [0, 15, 30, 45, 60, 75, 90]
    for rate in ReductionRateList:
        for method in MethodList:
            ResultList.append( regressionWithReduction(sampx, sampy, polyx, polyy, K, TestTimes, rate, method, choice) )
    #Mean Error
    x = array(ResultList).reshape(len(ReductionRateList), len(MethodList)).tolist()

    for row in x:
        for col in row:
            print str(col) + '\t',
        print

    for j in range(len(MethodList)):
        MeanError = []
        for i in range(len(ReductionRateList)):
            MeanError.append(x[i][j])
        drawMeanErrorDiagram(ReductionRateList, MeanError, 'Average MeanError, Method = ', MethodList[j])

def testWithLargeValue(sampx, sampy, polyx, polyy, K, MethodList, choice):
    N = len(sampx)
    Num = 5
    randlist = []
    for i in range(Num):
        r = randint(0, N-1)
        while(randlist.count(r) != 0):
            r = randint(0,N-1)
        randlist.append(r)
    for i in randlist:
        sampy[i] = str(float(sampy[i]) + 200)
    for method in MethodList:
        Regression(sampx, sampy, polyx, polyy, K, method, 'd', choice)

def testWithHigherK(sampx, sampy, polyx, polyy, K, MethodList, choice):
    for method in MethodList:
        Regression(sampx, sampy, polyx, polyy, K, method, 'e', choice)

def Regression(sampx, sampy, polyx, polyy, K, method, directory, choice):
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
    prediction = getPredictionValueList(polyx, theta, K)
    plt.figure().add_subplot(111).set_title(method ,fontsize = 18)
    plt.plot(toFloatList(polyx),prediction,'r-',label='prediction',linewidth=1)
    plt.plot(toFloatList(polyx),toFloatList(polyy),'g-',label='real',linewidth=0.8)
    plt.plot(toFloatList(sampx),toFloatList(sampy),'ko',label='sample')
    if(method == 'BR'):
        variance = getPredictionVarianceList(polyx, sigma, theta, K)
        add_variance = addList(prediction, variance)
        sub_variance = substractList(prediction, variance)
        plt.plot(toFloatList(polyx),add_variance,'b-', label = 'prediction + variance', linewidth = 1)
        plt.plot(toFloatList(polyx),sub_variance,'g--',  label = 'prediction - variance', linewidth = 2)
    plt.legend()
    if(choice == 'save'):
        plt.savefig('D:\\PA-1-data-text\\Result\\' + directory + '\\' + method + '.jpg', dpi=200)
    elif(choice == 'show'):
        plt.show()
    else:
        'Wrong'
    return prediction

def regressionWithReduction(sampx, sampy, polyx, polyy, K, TestTimes, ReducitionRate, method, choice):
    print str(K) + 'th ' + 'TestTimes = ' + str(TestTimes) + ' ReductionRate = ' + str(ReducitionRate) + '% Method = ' + method
    MeanErrorScalar = 0.0
    PredictionSum = zeros(len(polyx))
    for i in range(TestTimes):
        x = sampx[:]
        y = sampy[:]
        Theta, x, y = getThetaWithReduction(x, y, K, ReducitionRate, method)
        Prediction = getPredictionValueList(polyx, Theta, K)
        #PredictionSum += Prediction
        if(i == 0):
            drawPredictionDiagramWithReduction(x, y, polyx, polyy, Prediction, K, 'c', ReducitionRate, method, choice)
        MeanErrorVector = array(strlistToFloatvector(Prediction) - strlistToFloatvector(polyy))
        MeanErrorScalar += dot(MeanErrorVector.T, MeanErrorVector)

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
        r = randint(0, N-1)
        while(randlist.count(r) != 0):
            r = randint(0,N-1)
        randlist.append(r)

    for i in randlist:
        new_x.append(sampx[i])
        new_y.append(sampy[i])

    if(method == 'LS'):
        theta = LS(new_x, new_y, K)
    elif(method == 'RLS'):
        theta = RLS(new_x, new_y, K)
    elif(method == 'LASSO'):
        theta = LASSO(new_x, new_y, K)
    elif(method == 'RR'):
        theta = RR(new_x, new_y, K)
    elif(method == 'BR'):
        theta, variance = BR(new_x, new_y, K)
    else:
        print 'No method'
        return 0
    return theta, new_x, new_y
