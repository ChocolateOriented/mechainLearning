import operator
import math
import os
import numpy as np
import KNN


hwLabels= []
trainingFileList= os.listdir('trainingDigits')           # ① 获取目录内容
m=len(trainingFileList)
trainingMat=np.zeros((m,1024))
for i in range(m):
  # ② （以下三行）从文件名解析分类数字
  fileNameStr=trainingFileList[i]
  fileStr=fileNameStr.split('.')[0]
  classNumStr=int(fileStr.split('_')[0])
  hwLabels.append(classNumStr)
  trainingMat[i,:] =KNN.img2vector('trainingDigits/%s' %fileNameStr).append(classNumStr)

creatTrees(trainingMat,range(len(trainingMat[0])))
testFileList=os.listdir('testDigits')
errorCount=0.0
mTest=len(testFileList)
for i in range(mTest):
  fileNameStr=testFileList[i]
  fileStr=fileNameStr.split('.')[0]
  classNumStr=int(fileStr.split('_')[0])
  vectorUnderTest=KNN.img2vector('testDigits/%s' %fileNameStr)
  classifierResult=classify(vectorUnderTest)
  print ("the classifier came back with: %d,the real answer is: %d" \
         % (classifierResult,classNumStr))
  if(classifierResult!=classNumStr):
    errorCount+=1.0
    print(fileNameStr)
print ("\nthe total number of errors is: %d" %errorCount)
print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))


def majorityCnt(classList):
  classCount = {}
  for vote in classList:
    if vote not in classCount: classCount[vote] = 0
    classCount[vote] +=1
  sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),
                            reverse=True)
  return sortedClassCount[0][0]


def calcShannonEnt(dataset):
  labelCount = {}
  for featVec in dataset:
    curentLabel = featVec[-1]
    if curentLabel not in labelCount: labelCount[curentLabel] = 0
    labelCount[curentLabel] += 1
  shannonEnt = 0.0
  numEntrites = len(dataset)
  for lable in labelCount.keys():
    prob = float(labelCount[lable])/numEntrites
    shannonEnt -= prob*math.log(prob,2)
  return shannonEnt

def chooseBestFeatureSplit(dataset):
  # 计算基础信息熵
  numFeature = len(dataset[0])-1
  baseEntropy = calcShannonEnt(dataset)
  # 计算按不同特征值划分的集合的信息增益
  bestInfoGain = 0.0; beastFeature = -1
  for i in range(numFeature):
    featList = [e[i] for e in dataset]
    uniqueVals = set(featList)
    newEntropy = 0.0
    for val in uniqueVals:
      subDataSet = splitdataset(dataset,i,val)
      prob = len(subDataSet)/float(len(dataset))
      newEntropy += prob*calcShannonEnt(subDataSet)
    infoGain = baseEntropy - newEntropy
    if(infoGain > bestInfoGain):
      bestInfoGain = infoGain
      beastFeature = i
  return beastFeature


def splitdataset(dataset, axis, value):
  retDataSet = []
  for featueVec in dataset:
    if featueVec[axis] == value:
      reducedFeatVec = featueVec[:axis]
      reducedFeatVec.extend(featueVec[axis+1:])
      retDataSet.append(reducedFeatVec)
  return retDataSet



def creatTrees(dataset, featureLabels):
  classList = [e[-1] for e in dataset]
  # 叶子结点, 数据集中的数据类型一致
  if(classList.count(classList[0]) == len(classList)):
    return classList[0]
  # 如果所有的特征值都用过了, 返回出现次数最高的, 并带上概率
  if len(dataset[0]) == 1:
    return majorityCnt(classList)
  #继续划分
  bestFeat = chooseBestFeatureSplit(dataset)
  beatFeatLabels = featureLabels[bestFeat]
  myTree = {beatFeatLabels:{}}
  #删除该标签,使featureLabels顺序与splitDataSet一致
  del(featureLabels[bestFeat])
  featureValues = [e[bestFeat] for  e in dataset]
  uniqueValue = set(featureValues)
  for value in uniqueValue:
    subLabels = featureLabels[:]
    myTree[beatFeatLabels][value] = creatTrees(splitdataset(dataset,bestFeat,
                                                            value), subLabels)
  return myTree

def classify (inputTree, featLabels,testVec):
  firstStr = inputTree.keys()[0]
  featIndex = featLabels.index(firstStr)
  nextTree = inputTree[firstStr]
  key = testVec[firstStr]
  valueOfFeat = nextTree[key]
  if isinstance(valueOfFeat, dict):
    classLabel = classify(valueOfFeat,featLabels,testVec)
  else: classLabel = valueOfFeat
  return classLabel