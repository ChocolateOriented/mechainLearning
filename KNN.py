import numpy as np
import operator

a = np.tile([1,2,3], (3, 1))
print(a)

def createDataSet():
  group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
  labels = ['A', 'A', 'B', 'B']
  return group, labels


def classify0(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0]
  diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
  sqDiffMat = diffMat**2
  sqDistances = sqDiffMat.sum(axis=1)
