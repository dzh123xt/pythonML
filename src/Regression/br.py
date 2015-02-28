__author__ = 'dengzhihong'

from src.Regression.base import *

class BR(RegressionBase):
    @staticmethod
    def run(sampx, sampy, K):
        alpha = 0.6
        variance = 5
        y = RegressionBase.strlistToFloatvector(sampy)
        fai_matrix = RegressionBase.constructFaiMartix(sampx, K)
        sigma_theta_head = np.dot(fai_matrix, fai_matrix.transpose()) * (1.0/variance)
        sigma_theta_head += (1.0/alpha) * np.eye(sigma_theta_head.shape[0], sigma_theta_head.shape[1])
        sigma_theta_head = sigma_theta_head.I

        miu_theta_head = np.dot(sigma_theta_head,fai_matrix)
        miu_theta_head = np.dot(miu_theta_head, y) * (1.0/variance)

        return miu_theta_head, sigma_theta_head