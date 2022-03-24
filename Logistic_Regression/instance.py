from numpy import *
from Logistic_Regression import gradAscent
from Logistic_Regression import stocGradAscent


def classifyVector(inX, weights):
    prob = gradAscent.sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent.stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

#
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

multiTest()
'''
output:
the error rate of this test is: 0.328358
the error rate of this test is: 0.343284
the error rate of this test is: 0.373134
the error rate of this test is: 0.343284
the error rate of this test is: 0.432836
the error rate of this test is: 0.358209
the error rate of this test is: 0.343284
the error rate of this test is: 0.388060
the error rate of this test is: 0.343284
the error rate of this test is: 0.358209
'''