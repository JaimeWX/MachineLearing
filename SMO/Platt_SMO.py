# 运行过程总结1(对Platt_SMO_runResults1的分析)
'''
s1 smop()主while循环 循环退出条件：迭代次数超过指定的最大值 or 遍历整个集合都未对任意 alpha对 进行修改
    s2  smop()的第一个for循环：完整遍历(遍历所有的alpha值)
        s21 当i=0时，不满足selectJ() 中的 (len(validEcacheList)) > 1 (vel=[0]) ，所以 使用selectJ()中的 selectJrand()随机选择j:
            j=53 L==H fullSet, iter: 0 i:0, pairs changed 0
        s21 当i=1时，满足selectJ() 中的 (len(validEcacheList)) > 1  (vel=[0 1]), 所以 使用selectJ()中的最大化步长的方式获取j
            k=0 Ek=1.0 deltaE=0.0 因为不满足 deltaE > maxDelta(0)，所以返回的 maxK为初始值-1,Ej为初始值0 输出 L==H fullSet, iter: 0 i:1, pairs changed 0
        s22 当i=2时，满足selectJ() 中的 (len(validEcacheList)) > 1  (vel=[0 1 2]) 所以 使用selectJ()中的最大化步长的方式获取j
            k=0 Ek=1.0 deltaE=2.0 因为满足 deltaE > maxDelta(0) （注：k=1 Ek=1.0 deltaE=2.0） 所以返回的 maxK = 0 ，Ej = 1.0
                输出 fullSet, iter: 0 i:2, pairs changed 1 说明 第一对alpha值进行了改变！！！i = 2， j = 0
        s23 当i=3时，满足selectJ() 中的 (len(validEcacheList)) > 1  (vel=[0 1 2 3]) 所以 使用selectJ()中的最大化步长的方式获取j
            k=0 Ek=[[-2.22044605e-16]] deltaE=[[0.09242071]]
            k=1 Ek=[[-0.28954036]]     deltaE=[[0.38196107]]
            k=2 Ek=[[-2.22044605e-16]] deltaE=[[0.09242071]]
            当k=1时，deltaE的值最大，所以返回 maxK = 1，Ej = [[-0.28954036]]
                输出 L==H fullSet, iter: 0 i:3, pairs changed 1
        s24 当i=4时...
        s25 当i=5时，因为E5 = [[0.12342775]] labelMat[5] = 1 不满足下面的if语句
                if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0))
                所以输出为 fullSet, iter: 0 i:5, pairs changed 2，直接进入下一轮循环(i=6)
                ！！！ 需要特别强调是 vel 仍然等于 [0 1 2 3 4 ] ,不会加入5
                i = 6 与 i = 5 情况完全相同 所以 vel 仍然等于 [0 1 2 3 4 ]
                这意味着当 i = 7 时，vel = [0 1 2 3 4 7]，所以 k 只能分别等于0，1，2，3，4，从这5个数中寻求最大步长
        s26 ... 直到i = 99 结束第一个for循环
    s3  smop()的第二个for循环：遍历所有非边界alpha值(不在边界0或C上的值)
        s31 再结束了第一个for循环之后，entireSet = False，进入第二个for循环
        s32 找出所有的非边界点为 nonBoundIs = [ 0  3  4 17 18 25 46 55 94]
        s33 为每一个非边界点 从vel=[ 0  1  2  3  4  7  8 10 11 12 14 17 18 22 23 24 25 26 29 46 52 54 55 69 94 97]找到最大步长的配对alpha
                for example：i = 0时，k = 0，1，2，4，7... 返回的 maxK = 2 Ej = [[1.30436336]]
        s34 直到 i = 94 结束第二个for循环
                输出为 non-bound, iter: 1 i:94, pairs changed 0
            所以 alphaPairsChanged == 0 ，所以 entireSet = True，再次进入第一个for循环
    s4  smop()的第一个for循环的第二次执行
        s41 i：0-99 vel=[ 0  1  2  3  4  7  8 10 11 12 14 17 18 22 23 24 25 26 29 30 46 52 54 55 69 94 97] （给每个i找j）
        s42：当i=1...
        s43：当i=99时，输出为 fullSet, iter: 2 i:99, pairs changed 0
s1 alphaPairsChanged == 0 且 entireSet == False 不满足while循环条件，退出主循环while，结束程序
'''

# 运行过程总结2(对Platt_SMO_runResults2的分析)
'''
经过s1的大量计算之后，会得到9个不为0，91个为0的alpha值，这9个不为0的alpha值对应的样例即为支持向量
使用这9个支持向量去计算 w* = [[ 0.65307162] [-0.17196128]]
已知 w* 与 b* 即可对任意一个样例进行分类：
    ans = mat(ws) * datMat[0] + b 
        if ans > 0: 分类结果为 1 类
        if ans < 0: 分类结果为 -1 类  
'''

from numpy import *

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

# 用对象作为一个数据结构来保存所有重要的值
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 初始化1个mx2的矩阵，具体为：
        #   第一列：eCache是否有效的标志位
        #   第二列：实际的E值
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag

# 计算给定的  的决策函数和误差值
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 选择成对的第二个(内循环中的)的alpha值
def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    print("vel="+str(validEcacheList))
    if (len(validEcacheList)) > 1:
        print("sss")
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            print("k="+str(k))
            Ek = calcEk(oS, k)
            print("Ek="+str(Ek))
            deltaE = abs(Ei - Ek)
            print("deltaE="+str(deltaE))
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        print(maxK, Ej)
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    print("j="+str(j))
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    print("Ei="+str(Ei))
    # if语句的作用：如果误差很大，则需要对该样例所对应的alpha值进行优化
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        print("ooo")
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0


# C的详细解释
#   C是不同优化问题的权重
#   C一方面要保证所有样例的间隔不小于1.0，另一方面又要使分类间隔尽可能大，并且要在这两方面之间平衡
#   如果C很大，分类器会力图通过分隔超平面对所有的样例都正确分类
def smoP(dataMatIn, classLabels, C, toler, maxIter):
    # 构建一个数据结构来容纳所有的数据
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        print("hhh")
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                print("iA=" + str(i))
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            print("????????????????????????????????????")
            print(nonBoundIs)
            for i in nonBoundIs:
                print("iB="+str(i))
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

# 计算w
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    print(alphas)
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

dataArr,labelArr = loadDataSet('testSet.txt')
b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
ws = calcWs(alphas,dataArr,labelArr)
print(ws)



