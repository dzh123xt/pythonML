__author__ = 'dengzhihong'

from src.Regression.base import *
from scipy import optimize

class RR(RegressionBase):
    @staticmethod
    def run(sampx, sampy, K):
        y = RegressionBase.strlistToFloatvector(sampy)
        fai_matrix_trans = np.transpose(RegressionBase.constructFaiMartix(sampx, K))
        n = len(sampx)
        D = K + 1
        I_n = np.eye(n)
        A_matrix = np.array(np.row_stack( (np.column_stack( (-fai_matrix_trans,-I_n) ), np.column_stack( (fai_matrix_trans,-I_n) )) ))
        f_matrix = np.array(np.row_stack( ( np.zeros( (D,1) ), np.ones( (n,1) ) ) ))
        b_matrix = np.array(np.row_stack( (-y,y) ))
        # Arbitrary define value for theta and t
        x_matrix = np.array(np.row_stack( (np.ones( (D,1) ), np.ones((n,1)) ) ))

        def constraintFunc(x):
            #print '-----------------con--------------'
            #print "x : ",transpose(x)
            b_list = []
            c = b_matrix.tolist()
            for i in c:
                b_list.append(i[0])
            B = np.array(b_list)
            result = B - np.dot(A_matrix,x)
            return result

        def objFunc(x):
            #print '-----------------obj--------------'
            x = np.array(np.transpose(np.mat(x)))
            result = np.dot(np.transpose(f_matrix), x)
            #print "obj: ",float(result)
            return float(result)

        con = ({'type': 'ineq', 'fun': constraintFunc})
        res = optimize.minimize(objFunc, x_matrix, method='SLSQP', constraints=con)
        return np.transpose(np.mat(res.x[:D]))