__author__ = 'dengzhihong'

import matplotlib.pyplot as plt
from src.Methods.process_data import *

def showDiagram(x, y, title="", MethodName=""):
    ax = plt.figure().add_subplot(111)
    ax.set_title(title + '   Algorithm: '+ MethodName, fontsize = 18)
    plt.axis([-20,20,-20,20])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.plot(x, y,'b.')
    plt.show()
    
def showDiagramInCluster(data, Z, title=""):
    N = data.shape[0]
    X = np.array(data)
    ColorPattern = ['r.', 'g.', 'b.', 'y.', 'k.', 'm.', 'c.']

    ClusterX = [[],[],[],[]]
    ClusterY = [[],[],[],[]]
    for i in range(N):
        ClusterX[Z[i]].append(X[i][0])
        ClusterY[Z[i]].append(X[i][1])

    K = len(ClusterX)
    ax = plt.figure().add_subplot(111)
    ax.set_title(title  , fontsize = 18)
    plt.axis([-20,20,-20,20])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for k in range(K):
        plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
    plt.show()

def showLabeledDiagram(dataA_X, dataA_Y, title):
    X, Y = RawData2FloatXYList(dataA_X)
    Z = RawLabel2IntList(dataA_Y)
    N = len(Z)
    ColorPattern = ['r.', 'g.', 'b.', 'y.', 'k.', 'm.', 'c.']

    ClusterX = [[],[],[],[]]
    ClusterY = [[],[],[],[]]
    for i in range(N):
        ClusterX[Z[i]].append(X[i])
        ClusterY[Z[i]].append(Y[i])
    K = len(ClusterX)
    ax = plt.figure().add_subplot(111)
    ax.set_title('Real Labeled diagram of ' + title  , fontsize = 18)
    plt.axis([-20,20,-20,20])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for k in range(K):
        plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
    plt.show()
