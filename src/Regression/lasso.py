__author__ = 'dengzhihong'

from src.Regression.base import *
from scipy import optimize

class LASSO(RegressionBase):
    @staticmethod
    def run(sampx, sampy, K):
        y = RegressionBase.strlistToFloatvector(sampy)
        fai_matrix = RegressionBase.constructFaiMartix(sampx, K)
        product_fai = np.dot(fai_matrix, np.transpose(fai_matrix))
        n = len(sampx)
        D = K + 1
        Lambda = 0.18
        H_matrix = np.array(np.row_stack( (np.column_stack( (product_fai,-product_fai) ), np.column_stack( (-product_fai,product_fai) )) ))
        f_matrix = np.array(np.row_stack( (np.dot(fai_matrix,y), - np.dot(fai_matrix, y) ) ))
        f_matrix = -f_matrix + Lambda
        x_matrix = np.array(np.row_stack( (np.ones( (D,1) ), np.ones((D,1)) ) ))

        def constraintFunc(x):
            #print '-----------------con--------------'
            #print "x : ",transpose(x)
            return x

        def objFunc(x):
            #print '-----------------obj--------------'
            result = np.dot(np.dot(np.transpose(x), H_matrix), x)/2 + np.dot(np.transpose(f_matrix), x)
            #print "obj: ",float(result)
            return float(result)

        con = ({'type': 'ineq', 'fun': constraintFunc})
        res = optimize.minimize(objFunc, x_matrix, method='SLSQP', constraints=con)
        theta = []
        for i in range(res.x.shape[0]/2):
            theta.append(res.x[i] - res.x[i+res.x.shape[0]/2])
        return theta