# 用标准线性回归找到最佳拟合直线
#   w = (X.T * X).I * X.T * Y

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

# 计算最佳拟合直线(等同于求向量w(回归系数))
#   公式为 w = (X.T * X).I * X.T * Y
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # linalg.det() 求一个矩阵的行列式
    # 在此处求矩阵行列式的原因
    #   后续要求(X.T * X).I, 也就是要对矩阵求逆，如果行列式为0，则矩阵的逆不存在，会报错。
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # .I 求一个矩阵的逆
    ws = xTx.I * (xMat.T*yMat)
    return ws

xArr,yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr, yArr)

xMat = mat(xArr)
yMat = mat(yArr)
# yHat 使用回归系数与输入的内积计算出的预测值(回归方程)
yHat = xMat * ws
# 计算预测值yHat序列与真实值yMat序列的相关系数(计算分析出预测与真实的匹配程度)
print(corrcoef(yHat.T,yMat))
# 相关系数运行结果分析：
#   对角线元素为1.0,表示yMat与自己的匹配是最完美的
#   yHat和yMat的相关系数约为0.98
'''
    [[1.         0.98647356]
    [0.98647356 1.        ]]
'''

'''
# 画图
fig = plt.figure()
# 绘制训练集中所有的数据点
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
# 绘制最佳拟合直线
xCopy = xMat.copy()
#   将样例按升序排列(直线上的数据点次序不能混乱)
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:,1], yHat)
plt.show()
'''