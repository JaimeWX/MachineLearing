'''
梯度上升和随机梯度下降算法总结
    如何计算梯度的方向：计算出真实类别与预测类别的差值(使用sigmoid函数)，按照该差值的方向调整回归系数
    如何调整回归系数：w = w + alpha * error(gradient direction) * X(f(w)) [对应梯度上升法]
'''

from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 给每一个样例添加一个初始化为1(方便计算)的属性值，即每个样例有3个属性
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid函数
def sigmoid(inX):
    # 输入为 inX = w_0x_0 + w_1x_1 + w_2x_2
    # 返回一个范围在0～1之间的数值
    return 1.0/(1+exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha随着迭代次数的增加和i下标的增大，不断减小(但不会减小到0)
            #   在每次迭代时调整alpha值的目的是缓解所求回归系数在经过大量的迭代达到收敛的过程中的波动或高频震动
            '''
            "严格下降"
                双重for循环中，j为迭代次数，i为样例index
                当j<<max(i)时，alpha就不是严格下降的
            在优化算法中，应该避免参数的严格下降
            '''
            alpha = 4/(1.0+j+i)+0.0001
            # (算法核心) 从训练集中随机选取样本来更新回归系数
            #   这样做的目的是减少周期性的波动
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # for循环解释：为了在图上画出所有的样例点
    #   xcord1存储所有类别为1的样例的横坐标
    #   ycord1存储所有类别为1的样例的纵坐标
    #   xcord2存储所有类别为0的样例的横坐标
    #   ycord2存储所有类别为0的样例的纵坐标
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # (核心代码) 画出最佳拟合直线 60个点的对应关系
    x = arange(-3.0, 3.0, 0.1)
    # 由 w_0x_0 + w_1x_1 + w_2x_2 = 0 推出y点的坐标(下一行代码) 注：x_0 = 1
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

'''
dataArr,labelMat = loadDataSet()
weights = stocGradAscent1(array(dataArr),labelMat)
plotBestFit(weights)
'''