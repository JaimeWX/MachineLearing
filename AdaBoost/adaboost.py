from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# 单层决策树(decision stump)def1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    # 'lt' is 'less than'
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

# 单层决策树def2
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T # .T 转置(把1X5转换为5X1)
    # m,n分别为5,2. 意为datMat是一个5X2的矩阵
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):
        # 第一层for循环 i为0，1
        # 当i为0时，rangeMin表示在第一个属性的5个属性值[[1. ] [2. ] [1.3] [1. ] [2. ]]找到最小的值1. rangeMax is 2.
        # 当i为1时，rangeMin表示在第二个属性的5个属性值[[2.1] [1.1] [1. ] [1. ] [1. ]]找到最小的值1. rangeMax is 2.1
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        # 通过计算数据集中所有属性值的最大值和最小值来了解需要多大的步长
        # 使用stepSize来调整阈值的变化范围
        stepSize = (rangeMax-rangeMin)/numSteps # (0.1 and 0.11)
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
            # 三层for循环 48 = 2(0,1)x12(-1~10)x2('lt','gt')
            # 三层for循环详细解释：(数据集的第0维为[[1] [2] [1.3] [1] [2]] 第1维为[[2.1] [1.1] [1] [1] [1]])
                # 以阈值为 1.0 为例.
                # lt: 预测值为: 第0维[[-1] [ 1] [ 1] [-1] [ 1]]   错误率为[[1] [0] [1] [0] [0]] 加权错误率为0.4
                # gt: 预测值为: 第0维[[ 1] [ -1] [ -1] [1] [ -1]] 错误率为[[0] [1] [0] [1] [1]] 加权错误率为0.6
            # 三层for循环最终的输出结果为
                # 最佳阈值为1.3 选用的最佳维数为0 选用的分类条件为'lt' 最小错误率为0.2
                # lt: 预测值为: 第0维[[-1] [ 1] [ -1] [-1] [ 1]] (只有第一个属性值的预测出错)
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

# adaBoost
def adaBoostTrainDS(dataArr,classLabels,numIt=40): # numIt 迭代次数(需要用户指定的参数)(如果在某次迭代之后错误率为0，则会退出迭代过程)
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    # aggClassEst 记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))
    # adaBoost详细解释(for循环的运行)
    # 第一次迭代
    #       第一个bestStump(bestStump1)的error为0.2, alpha为0.69
    #       回顾：classLables = [[1] [1] [-1] [-1] [1]] classEst(bestStump1的分类结果) = [[-1] [1] [-1] [-1] [1]]
    #       由此可知，只有第一个样例分类错误，所以通过计算更新所有样例的权重D:
    #       初始为D = [[0.2] [0.2] [0.2] [0.2] [0.2]] 更新后为D1 = [[0.5] [0.125] [0.125] [0.125] [0.125]]
    #       aggErrors = [[1] [0] [0] [0] [0]] errorRate = 0.2(这两个值都所属于多分类器系统)
    # 第二次迭代 即为D改变之后，重新训练出的bestStump(bestStump2)
    #       bestStump2: [[1] [1] [-1] [-1] [-1]] error = 0.125 alpha = 0.97 (error与D关系密切，可不是0.2)
    #       D2 = [[0.28571429] [0.07142857] [0.07142857] [0.07142857] [0.5]]
    #       aggErrors = [[0] [0] [0] [0] [1]] errorRate = 0.2
    # 第三次迭代
    #       bestStump3: [[1] [1] [1] [1] [1]] error = 0.1428 alpha = 0.89 (error与D关系密切，可不是0.4)
    #       D3 = [[0.16666667] [0.04166667] [0.25      ] [0.25      ] [0.29166667]]
    #       aggErrors = [[0] [0] [0] [0] [0]] errorRate = 0
    # 特别说明：(aggErrors 中的0表示分类正确，1表示分类错误)
    for i in range(numIt):
        # build decision stump
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        # alpha 会告诉总分类器本次单层决策树输出结果的权重
        # max(error,1e-16) 确保在没有错误时不会发生除0溢出
        # alpha计算公式：1/2ln(1-error/error)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        # 把此bestStump(基分类器)加入到多分类器系统中
        weakClassArr.append(bestStump)
        # 规范化因子
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        # sign(x): x>0 -> x=1; x=0 -> x=0; x<0 -> x=-1
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break
    # !!! 为了测试adaClassify(datToClass,classifierArr)，则返回值不能有aggClassEst
    return weakClassArr #,aggClassEst

# 测试 对datMat中没有出现的样例进行分类 例如[0,0],[5,5]
def adaClassify(datToClass,classifierArr):
    # dataToClass 为一个或多个多个待分类样例
    # classifierArr 为多个弱分类器组成的数组
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
    return sign(aggClassEst)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


#datMat,classLabels = loadSimpData()
# 权重向量D：给datMat中的5个样例初始化相同的权重(均为0.2)
# D = mat(ones((5,1))/5)
# print(buildStump(datMat,classLabels,D))
#classifierArray = adaBoostTrainDS(datMat,classLabels,9)
# print(classifierArray)

# 对adaClassify(datToClass,classifierArr)的执行
# datArr,labelArr = loadSimpData()
# classifierArr = adaBoostTrainDS(datArr,labelArr,30)
# print(adaClassify([[0,0],[5,5]],classifierArr))

# 实战测试
datArr,labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray = adaBoostTrainDS(datArr,labelArr,50)  # 50个弱分类器
testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr,classifierArray)
# 67是因为测试集有67个样例
errArr = mat(ones((67,1)))
print(errArr[prediction10 != mat(testLabelArr).T].sum())

