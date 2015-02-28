__author__ = 'dengzhihong'

from src.Regression.base import *

class LS(RegressionBase):
    @staticmethod
    def run(sampx, sampy, K):
        y = RegressionBase.strlistToFloatvector(sampy)
        fai_matrix = RegressionBase.constructFaiMartix(sampx, K)
        Theta = np.dot(fai_matrix, fai_matrix.transpose())
        Theta = Theta.I
        Theta = np.dot(Theta, fai_matrix)
        Theta = np.dot(Theta, y)
        return Theta