# 用局部加权线性回归找到最佳拟合直线
#   w = (X.T * W * X).I * X.T * W * Y (每个样例都有一个属于它的对角权重矩阵，每个样都有自己的回归系数向量)

# 为什么在已有标准线性回归的情况下，还要有局部加权线性回归?
#   标归求的是具有最小均方误差的无偏估计，可能出现欠拟合
#   局归允许在估计中引入一些偏差，从而降低预测的均方误差

# 标准线性回归与局部加权线性回归的区别
#   标归(直线建模)：每一个样例拥有相同的回归系数
#   局归(非直线建模)：每一个样例用于不同的回归系数

# 高斯核中参数k的取值的影响
#   k = 1.0，如同将所有的数据视为等权重，其结果与标准的回归一致(可能会欠拟合)
#   k = 0.01 效果很好，抓住了数据的潜在模式
#   k = 0.003 纳入了太多的噪声点，拟合的直线与数据点过于贴近(过拟合)

'''
需要特别说明的地方
    高斯核原公式中是 abs(x_i - X)，但是实际代码使用了 pow(x_i - X) # X表示训练数据集
'''

from numpy import *
import matplotlib.pyplot as plt

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

# 得到数据集中一个样例的预测值
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # weights 一个mxm(方阵(阶数等于样本点个数))的对角线元素为1，其余元素为0的对角权重矩阵 等价于 weights给每个样例初始化了一个权重
    #   numpy.eye(N,M=None,k=0,dtype=<class ‘float’>,order=‘C) 返回的是一个二维的数组(N,M)，对角线的地方为1，其余的地方为0.
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]
        # 使用了高斯核来对待预测点附近的点赋予更高的权重
        # 随着样本点与待预测权重距离的递增，权重将以指数级衰减
        # 用户指定参数k的作用：控制衰减的速度
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
        '''
            运行过程分析
                对于每一个待预测样例，都有一个属于它的对角权重矩阵，表示数据集中其它样例与它的欧式距离，距离越近的点，权重越大
        '''
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 得到数据集中所有样例的预测值
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


'''
xArr,yArr = loadDataSet('ex0.txt')
# 求数据集中所有样例的预测值
yHat = lwlrTest(xArr,xArr, yArr,0.01)
'''


'''
# 画图
xMat = mat(xArr)
srtInd = xMat[:,1].argsort(0)
xSort = xMat[srtInd][:,0,:]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T[:,0].flatten().A[0],s=2,c='red')
plt.show()
'''

# 计算预测误差的大小
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

abX,abY = loadDataSet('abalone.txt')
# 比较取不同的k值，在训练集上的误差大小
# 运行结果
#   k = 0.1 56.78868743048742
#   k = 1 429.8905618704253
#   k = 10 549.1181708829058
yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
print(rssError(abY[0:99],yHat01.T))
print(rssError(abY[0:99],yHat1.T))
print(rssError(abY[0:99],yHat10.T))
# 比较取不同的k值，在训练集上的误差大小
# 运行结果
#   k = 0.1 57913.51550155909
#   k = 1 573.5261441895273
#   k = 10 517.571190538078
yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
print(rssError(abY[100:199],yHat01.T))
print(rssError(abY[100:199],yHat1.T))
print(rssError(abY[100:199],yHat10.T))

'''
运行结果分析
    局部加权线性回归 
        k = 10 训练误差最大但测试误差最小
        k = 0.1 训练误差最小但测试误差最大
    标准线性回归
        同样使用abX[0:99]作为训练集，使用abY[100:199]作为测试集
            预测误差为518.6363
        可见标准线性回归与局部加权线性回归达到了类似的效果
    (必须在未知数据上比较效果才能选取最佳模型)
'''
'''
Q: k = 10 是最佳核吗
A: 或许是(应该在不同的样本集做10次测试来比较结果，此处只做了一次)
'''


