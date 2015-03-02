__author__ = 'dengzhihong'

from src.Regression.base import *
from scipy import optimize
from numpy import *

class RR(RegressionBase):
    @staticmethod
    def run(sampx, sampy, K):
        y = RegressionBase.strlistToFloatvector(sampy)
        fai_matrix_trans = transpose(RegressionBase.constructFaiMartix(sampx, K))
        n = len(sampx)
        D = K + 1
        I_n = eye(n)
        A_matrix = array(row_stack( (column_stack( (-fai_matrix_trans,-I_n) ), column_stack( (fai_matrix_trans,-I_n) )) ))
        f_matrix = array(row_stack( ( zeros( (D,1) ), ones( (n,1) ) ) ))
        b_matrix = array(row_stack( (-y,y) ))
        # Arbitrary define value for theta and t
        x_matrix = array(row_stack( (ones( (D,1) ), ones((n,1)) ) ))

        def constraintFunc(x):
            #print '-----------------con--------------'
            #print "x : ",transpose(x)
            b_list = []
            c = b_matrix.tolist()
            for i in c:
                b_list.append(i[0])
            B = array(b_list)
            result = B - dot(A_matrix,x)
            return result

        def objFunc(x):
            #print '-----------------obj--------------'
            x = array(transpose(mat(x)))
            result = dot(transpose(f_matrix), x)
            #print "obj: ",float(result)
            return float(result)

        con = ({'type': 'ineq', 'fun': constraintFunc})
        res = optimize.minimize(objFunc, x_matrix, method='SLSQP', constraints=con)
        return transpose(mat(res.x[:D]))