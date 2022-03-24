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

# 梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    # mat()作用：转换成为Numpy matrix
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    # alpha为向目标移动的步长
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        # dataMatrix*weights 为 sigmoid函数的输入，即 z = w_0x_0 + w_1x_1 + w_2x_2
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        # (算法核心) 梯度上升算法用于求函数的最大值
        # 迭代公式：w = w + aplha * gradient_w * f(w)
        # 此处按照真实类别与预测类别的差值的方向调整回归系数
        # (dataMatrix.transpose()是一个3x100的矩阵，error是一个有100个元素的列向量,相乘则为有3个元素的向量）
        weights = weights + alpha * dataMatrix.transpose() * error

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
weights = gradAscent(dataArr,labelMat)
print(weights)
plotBestFit(weights.get A())
'''