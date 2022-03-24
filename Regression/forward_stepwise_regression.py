'''
前向逐步线性回归算法总结
    算法核心：贪心算法，即每一步都尽可能减少误差。
             一开始，所有的权重都设为0，然后每一步所做的决策都是对某个权重增加或减少一个很小的值
    关键问题：每一步如何减小误差？(3重for循环分析) [结合abalone.txt中的数据集] [运行结果见forward_stepwise_regression_runResults]
        ws = [[0. 0. 0. 0. 0. 0. 0. 0.]] 为初始回归系数，lowestError 初始化为正无穷
        每一次迭代对于每一个特征所对应的回归系数，根据步长去增大或减少该回归系数，然后根据新的ws去计算预测误差，如果比lowestError小，则将
        这个新的ws设置为ws*
        举例说明：
            第一轮循环：
                对第4个特征前的回归系数增加0.01,求得的误差相比增加或减少其他特征前的回归系数最小(也包括对第4个特征前的回归系数减少0.01)
                所以第一轮结束后 ws = [[0.    0.    0.    0.001 0.    0.    0.    0.   ]]
            第二轮循环：
                在 ws = [[0.    0.    0.    0.001 0.    0.    0.    0.   ]]的基础上 重复
            ...
    步长过大导致的问题：系数可能会饱和并在特定值之间来回震荡(stepSize = 0.01, numIt = 200时，第一个权重在0.04和0.05之间来回震荡)
    逐步线性回归算法的好处：帮助理解现有的模型并作出改进
        当构建一个模型之后，可以运行该算法找出重要特征，这样可能及时停止对那些不重要的特征的收集
        争议性示例
            stepSize = 0.01, numIt = 200时，第一个和第六个权重都一直为0，即表明它们不对目标值造成任何影响，这些特征可能不重要
            stepSize = 0.001, numIt = 5000时，第一个和第六个权重均不为0
'''

from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 计算预测误差的大小
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

# 标准化数据(所有特征都减去各自的均值并除以方差)
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    # 再次强调 mean(inMat,0) 或 var(inMat,0) 中的0表示输出的不是一个数而是一个1xn的矩阵
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100): # eps表示每次迭代需要调整的步长 numIt表示迭代次数
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n=shape(xMat)
    # returnMat = zeros((numIt,n))
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            print(j)
            # for sign in [-1,1]的解释：sign只能为-1，1 如果将[-1,1]换位[-3,3]，则sign只能为-3，3
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                print(wsTest)
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                print(rssE)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        # returnMat[i,:]=ws.T
    return ws

xArr,yArr = loadDataSet('abalone.txt')
print(stageWise(xArr, yArr,0.001,100))