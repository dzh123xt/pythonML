__author__ = 'dengzhihong'

from src.Regression.base import *

class RLS(RegressionBase):
    @staticmethod
    def run(sampx, sampy, K):
        Lambda = 0.235
        y = RegressionBase.strlistToFloatvector(sampy)
        fai_matrix = RegressionBase.constructFaiMartix(sampx, K)
        Theta = np.dot(fai_matrix, fai_matrix.transpose())
        L = Lambda * np.eye(Theta.shape[0], Theta.shape[1])
        Theta += L
        Theta = Theta.I
        Theta = np.dot(Theta, fai_matrix)
        Theta = np.dot(Theta, y)
        return Theta