__author__ = 'dengzhihong'

import matplotlib.pyplot as plt

def drawDiagram(x, y, title="", directory="", MethodName=""):
    ax = plt.figure().add_subplot(111)
    ax.set_title(title + '   Algorithm: '+ MethodName, fontsize = 18)
    plt.axis([-20,20,-20,20])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.plot(x, y,'b.')
    plt.savefig('./Diagram/' + directory + '/' + MethodName + '.jpg', dpi=200)

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
    ax.set_title('Real clustering of ' + title  , fontsize = 18)
    plt.axis([-20,20,-20,20])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for k in range(K):
        #index = Z[i] - 1
        plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
    plt.show()

def drawDiagramInCluster(X, Z, Title="", MethodName="", Directory=""):
    D = X.shape[2]
    N = X.shape[0]
    K = Z.shape[1]
    Class = []
    ColorPattern = ['r.', 'g.', 'b.', 'y.', 'k.', 'm.', 'c.', 'ro']
    for i in range(N):
        for k in range(K):
            if(Z[i][k] == 1.0):
                Class.append(k)
                break
    ClusterX = [[],[],[],[]]
    ClusterY = [[],[],[],[]]
    for i in range(N):
        ClusterX[Class[i]].append(X[i][0][0])
        ClusterY[Class[i]].append(X[i][0][1])

    ax = plt.figure().add_subplot(111)
    ax.set_title(Title + '   Algorithm: '+ MethodName, fontsize = 18)
    plt.axis([-20,20,-20,20])
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for k in range(K):
        plt.plot(ClusterX[k], ClusterY[k], ColorPattern[k], label = 'c' + str(k))
    plt.legend()
    plt.savefig('./Diagram/' + Directory + '/' + MethodName + Title +'.jpg', dpi=200)

def drawLabeledDiagram(dataA_X, dataA_Y, title):
    x, y = getCoordFromList(dataA_X)
    Z = str2IntList(dataA_Y)
    showDiagramInCluster(x, y, Z, title)
