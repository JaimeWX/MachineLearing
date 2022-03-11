from numpy import *
# 算法的几个核心点：
#   决策函数fXi的计算
#   for循环中的if语句，根据Ei判断是否该对alpha[i]进行优化
#   L,H的计算
#   eta的计算
#   优化后的alpha[j]与alpha[i]的计算
#   b的计算

# 举例说明(simpleSMO_runResult)：
#   当 i = 0, alpha = 0, b = 0时, fX1 = 0，E0 = 1. 则 labelMat[0]*E0 = -1，-1 < 0.001 and 0 < 0.6 为 true
#   则需要对第0个样例对应的alpha值(0)进行优化
#   假设选中的另一个样例j为77, E77 = -1
#   L = 0, H = 0.6(y0=-1,y77=1)
#   解得 eta = [[-23.730183]]
#   计算alpha[j]的新值(区别于原来的alpha[j] = 0) = [[0.08428085]]
#   可见alpha[j]发生了轻分析化
#   计算alpha[i]的新值(区别于原来的alpha[i] = 0) = [[0.08428085]]
#   解得 b = [[-2.3842744]]   （到这一步则成功改变了一对alpha的值！！！）
# 对simpleSMO_runResult的进一步分析：
#   i = 0(j = 77)执行完成之后，i=1，但不符合if条件，i=2（j=32）导致L==H，...，i=5（j=20）导致j只发生轻微变化...i=99(j=15)
#   此次for循环产生了3对新的alpha[i]与alpha[j] alphaPairsChanged == 3,所以 iter仍为0，继续进行for循环从i=0开始...直到满足条件退出while循环
#   最终 b = [[-3.88819879]]
# 3个支持向量分别为
#   [4.658191, 3.507396] -1.0
#   [3.457096, -0.082216] -1.0
#   [6.080573, 0.418886] 1.0

# 打开文件并对其进行逐行解析，从而得到每行的类别标签和整个数据矩阵
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while (j==i):
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter): # 常数C, 容错率toler, 退出当前最大的循环次数
    # 在此处 mat()的作用是将数组转换成矩阵；transpose()的作用是将 1xn 矩阵转换成 nx1 矩阵(转置)
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    # m,n 为矩阵的行列数
    b = 0; m,n = shape(dataMatrix)
    # 初始化一个 mx1 的，值为0 的alpha矩阵(每一个样例对应一个alpha)
    alphas = mat(zeros((m,1)))
    # iter存储在没有任何alpha改变的情况下遍历数据集的次数
    iter = 0
    # 外循环
    while (iter < maxIter):
        alphaPairsChanged = 0
        print("BBB")
        # 内循环
        for i in range(m):
            print("i="+str(i))
            #print("i="+ str(i))
            # fXi为决策函数，即所预测的类别
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #print(fXi)
            # Ei 为预测结果与真实结果的误差
            Ei = fXi - float(labelMat[i]) #if checks if an example violates KKT conditions
            # if语句的作用：如果误差很大，则需要对该样例所对应的alpha值进行优化             
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # SMO算法对成对的alpha值进行优化，所以需要再从dataArr中随机找出一个样例j，并计算j的预测结果和误差
                j = selectJrand(i,m)
                print("j="+str(j))
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 计算L与H，其作用为调整alpha[j]到0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue # 再次强调continue，其作用为结束本次循环，直接运行下一次for循环
                # eta为alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                print(eta)
                if eta >= 0: print("eta>=0"); continue
                # 计算新的alphas[j]的值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 检查alpha[j]是否有轻微的变化
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                # 更新alpha[i]的值，其大小与alpha[j]相同，但方向相反(一个增加，另一个减小)
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 给优化之后的alpha[i]与alpha[j]分别设置新的常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        # 检查alpha的值是否做了更新，这样做的目的是：只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas


dataArr,labelArr = loadDataSet('testSet.txt')
b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
print(b)
for i in range(100):
    if alphas[i] > 0.0:
        print(dataArr[i],labelArr[i])