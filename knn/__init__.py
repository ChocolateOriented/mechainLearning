import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import KNN
import os
'''

KNN.handwritingClassTest()


KNN.datingClassTest()
datingDataMat, datingLabels = KNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*np.array(
    datingLabels), 15.0*np.array(datingLabels))
plt.show()


x_values = list(range(1,5000))
y_values = [x**3 for x in x_values]
plt.scatter(x_values,y_values,c=y_values,cmap=plt.cm.Blues,edgecolors='none',
            s=40)
plt.show()
'''

def handwritingClassTest():
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
    trainingMat[i,:] =img2vector('trainingDigits/%s' %fileNameStr)
  testFileList=os.listdir('testDigits')
  errorCount=0.0
  mTest=len(testFileList)
  for i in range(mTest):
    fileNameStr=testFileList[i]
    fileStr=fileNameStr.split('.')[0]
    classNumStr=int(fileStr.split('_')[0])
    vectorUnderTest=img2vector('testDigits/%s' %fileNameStr)
    classifierResult=classify0(vectorUnderTest, \
                               trainingMat,hwLabels,15)
    print ("the classifier came back with: %d,the real answer is: %d" \
           % (classifierResult,classNumStr))
    if(classifierResult!=classNumStr):
      errorCount+=1.0
      print(fileNameStr)
  print ("\nthe total number of errors is: %d" %errorCount)
  print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))