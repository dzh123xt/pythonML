__author__ = 'dengzhihong'

from src.Methods.regression_methods import *

if __name__ == '__main__':
    sampx = str(open('../data/RegressionData/polydata_data_sampx.txt', 'r').read()).split()
    sampy = str(open('../data/RegressionData/polydata_data_sampy.txt', 'r').read()).split()
    polyx = str(open('../data/RegressionData/polydata_data_polyx.txt', 'r').read()).split()
    polyy = str(open('../data/RegressionData/polydata_data_polyy.txt', 'r').read()).split()

    K = 5
    MethodList = ['LS','RLS','LASSO','RR','BR']

    # problem b
    testWithRegression(sampx, sampy, polyx, polyy, K, MethodList, 'show')
    # problem c
    testWithReduction(sampx, sampy, polyx, polyy, K, MethodList, 'show')
    # problem d
    testWithLargeValue(sampx, sampy, polyx, polyy, K, MethodList, 'show')
    # problem e
    K = 20
    testWithHigherK(sampx, sampy, polyx, polyy, K, MethodList, 'save')