__author__ = 'dengzhihong'

import math
import numpy as np

def multivariateGaussian(vector_x, vector_mean, mat_covariance):
    k = vector_x.shape[0]
    first = 1 / math.sqrt((2 * math.pi * np.linalg.det(mat_covariance))**k)
    sub = vector_x - vector_mean
    #norm = -0.5 * dot( dot( transpose(sub), inv(mat_covariance)),  sub)
    norm = -0.5 * np.dot( np.dot( sub, np.linalg.inv(mat_covariance)),  np.transpose(sub))
    result = first * pow(math.e, float(norm))
    return result

def getSquareNorm(X, Mean):
    sub = np.transpose(X) - np.transpose(Mean)
    result = np.dot( np.transpose(sub), sub)
    return float(result)