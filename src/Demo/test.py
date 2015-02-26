__author__ = 'dengzhihong'

from src.Cluster.base import *
from src.Cluster.kmeans import *
from src.Cluster.em import *

if __name__ == '__main__':
    dataA_X = str(open('../data/ClusterData/cluster_data_dataA_X.txt', 'r').read()).split()
    dataA_Y = str(open('../data/ClusterData/cluster_data_dataA_Y.txt', 'r').read()).split()
    dataB_X = str(open('../data/ClusterData/cluster_data_dataB_X.txt', 'r').read()).split()
    dataB_Y = str(open('../data/ClusterData/cluster_data_dataB_Y.txt', 'r').read()).split()
    dataC_X = str(open('../data/ClusterData/cluster_data_dataC_X.txt', 'r').read()).split()
    dataC_Y = str(open('../data/ClusterData/cluster_data_dataC_Y.txt', 'r').read()).split()

    #showLabeledDiagram(dataA_X, dataA_Y, 'Data A')
    #showLabeledDiagram(dataB_X, dataB_Y, 'Data B')
    #showLabeledDiagram(dataC_X, dataC_Y, 'Data C')

    #Kmeans-----------------------------------------------------------------

    #KMeans.testWithKmeans(dataA_X, 'Data A', 4)
    #KMeans.testWithKmeans(dataB_X, 'Data B', 4)
    #KMeans.testWithKmeans(dataC_X, 'Data C', 4)

    #EM---------------------------------------------------------------------

    EM.testWithEM(dataA_X, 'Data A', 4)
    #EM.testWithEM(dataB_X, 'Data B', 4)
    #EM.testWithEM(dataC_X, 'Data C', 4)

    #Mean-shift-------------------------------------------------------------
    #testWithMeanShift(dataA_X, 'Data A')
    #testWithMeanShift(dataB_X, 'Data B')
    #testWithMeanShift(dataC_X, 'Data C')


