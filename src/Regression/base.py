__author__ = 'dengzhihong'

import numpy as np

class RegressionBase(object):

    # This method get a list of data in string form and turn them into float vector
    @staticmethod
    def strlistToFloatvector(strlist):
        floatvector = []
        for i in range(0, len(strlist)):
            floatvector.append(float(strlist[i]))
        floatvector = np.transpose(np.mat(floatvector))
        return floatvector

    # This method get a single float scalar value x and get its fai(x) float list not vector
    @staticmethod
    def getFaiList(scalar_x, K):
        floatlist_fai = []
        for i in range(0, K+1):
             floatlist_fai.append(scalar_x**i)
        return floatlist_fai

    # This method will construct Fai matrix which contains many fai vector of x
    @staticmethod
    def constructFaiMartix(strlist_x, K):
        fai_martix = []
        for i in range(0,len(strlist_x)):
            float_x = float(strlist_x[i])
            fai_martix.append(RegressionBase.getFaiList(float_x, K))
        FaiMartix = np.transpose(np.mat(fai_martix))
        return FaiMartix

    # This method get the float x and get its prediction float value y
    @staticmethod
    def getPredictionValue(star_scalar_x, theta, K):
        Fai = np.mat(RegressionBase.getFaiList(star_scalar_x, K))
        return float(np.dot(Fai,theta))

    # This method get a string list x and get the float list prediction value y
    @staticmethod
    def getPredictionValueList(star_list_x, theta, K):
        x = RegressionBase.toFloatList(star_list_x)
        y = []
        for element in x:
            y.append(RegressionBase.getPredictionValue(element, theta, K))
        return y

    # This method will turn string list to float list, which is highly useful
    @staticmethod
    def toFloatList(stringlist):
        floatlist = []
        for i in range(0, len(stringlist)):
             floatlist.append(float(stringlist[i]))
        return floatlist

    @staticmethod
    def run(sampx, sampy, K):
        pass

