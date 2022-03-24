'''
岭回归算法总结
    需要解决的问题：当特征比样样例还多(n>m)，即输入数据的矩阵X不是满秩矩阵，而非满秩矩阵会在求逆时出现问题
         (求回归系数时需要求 (X.TxT).I)
    解决问题的办法：(X.TxX) -> (X.TxX + λI) 使得矩阵非奇异，从而能够求逆(不可逆的矩阵称为奇异矩阵)
                    I: mxm的单位矩阵(X为mxn)
                    λ：用户指定的数值
    回归系数的计算公式：(X.TxX + λI).I x X.TxY
    缩减：通过引入λ去限制所有w之和(通过引入惩罚项(λ)，从而减少不重要的参数)
         应用缩减法模型会增加偏差的同时减小模型的方差
    如何得到λ：预测误差最小化
        数据获取之后，首先抽一部分数据用于测试，剩余的作为训练集用于训练参数w
        训练完毕后在测试集上测试预测性能
        通过选取不同的λ来重复上述测试过程，最终得到一个使预测误差最小的λ
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

# 计算回归系数
def ridgeRegres(xMat, yMat, lam=0.2): # lam 即代表lambda(lambda是关键字，直接用会报错)
    xTx = xMat.T * xMat
    # eye(shape(xMat)[1]) 构建一个与xMat列数相同的nxn单位矩阵
    denom = xTx + eye(shape(xMat)[1]) * lam
    # 检查行列式为0的原因：当 lanmda = 0 时，矩阵求逆仍可能会出现错误(denom.I)
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

# 在一组λ上测试结果
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    '''
    对特征做标准化处理
        目的：使每维特征具有相同的重要性
        原因：为了使用岭回归和缩减技术
    '''
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    # numpy.var(arr, axis = None)：计算指定数据(数组元素)沿指定轴(如果有)的方差
    #   axis = 0 表示沿列的方差
    #   axis = 1 表示沿行的方差
    '''
    对特征做标准化处理的具体做法
        所有特征都减去各自的均值并除以方差
    '''
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    # 在30个不同的λ下调用ridgeRegres()
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        # exp(x) 返回e的x方
        # λ呈指数级变化，这样做的目的是可看出λ非常大或非常小时对结果造成的影响
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

abX,abY = loadDataSet('abalone.txt')
ridgeWeights = ridgeTest(abX,abY)
print(ridgeWeights)

'''
#画图展示缩减效果
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()
'''
