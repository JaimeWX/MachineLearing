'''
算法总结
    Q1：为什么在K-Means的基础上，产生了Binary_K-Means?
    A1: K-Means 只是收敛到了局部最小，而非全局最小

    运行过程总结(伪代码)(运行结果对应BKM_runResults)
    s1 把所有样本加入一个簇中(根据所有样本所产生的均值作为这个簇的聚类中心),并把该聚类中心加入一个列表cenlist中
    s2 计算这唯一一个簇中每一个样本的误差值(样本与聚类中心欧式距离的平方值)，并把每一个样本所属的簇的索引值(均为0)与误差值加入到一个矩阵clusterAssment中
    s3 对cenlist中的每一个簇
            使用K-Means算法(k指定为2)，把这个簇分为2个簇，计算这次划分的总误差(在这两个簇的所有样例的总误差+不在这两个簇中的所有样例的总误差)
            centroidMat, splitClustAss 分别为 这次划分所形成包含2个簇的聚类中心的列表 和 索引值与误差值所构成的矩阵
            如果这次划分的总误差小于最小误差(初始为无穷大)，则认为这次划分是有效的，更新 这次划分为最佳划分bestCentToSplit
                                                                             最佳划分聚类中心bestNewCents的列表为centroidMat
                                                                             最佳划分索引值与误差值所构成的矩阵bestClustAss为splitClustAss.copy()
                                                                             最小误差为这次划分的总误差
    # 详细说明s3的原因：解释清楚第二轮for循环i分别等于0和1时的运行过程
    s4 根据s3的最佳划分结果更新cenlist和clusterAssment，直到满足用户指定的簇数目

    难点1 nonzero()的使用
    难点2 使用2个数组过滤器去更新簇分类结果

    对BKM_runResults的分析
        第一轮for循环：使用i = 0，去更新cenlist和clusterAssment
        第二轮for循环：使用i = 0，去更新cenlist和clusterAssment,原因是i = 0 的总误差小于i = 1的总误差
        cenlist的变化：
            第一轮for循环结束后：[[-1.7035159500000003, 0.27408125000000005], [2.93386365, 3.12782785]]
            第二轮for循环结束后：[[-0.45965614999999993, -2.7782156000000002], [2.93386365, 3.12782785], [-2.94737575, 3.3263781000000003]]
                第二轮for循环i=0产生的centroidMat：[[-0.45965615 -2.7782156 ] [-2.94737575  3.3263781 ]]
                第二轮for循环i=1产生的centroidMat：[[3.6690305  4.03686067] [2.61879214 2.73824236]]
'''

from numpy import *
# 无监督学习，只有属性值，没有类别标签
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map() 会根据提供的函数对指定序列做映射。在python3中返回的是迭代器(内存地址)，可以使用list()转换成列表，看到具体的数值
        fltLine = list(map(float ,curLine)) # 这里的list是我加的，原因是为了更好地看到输出结果
        dataMat.append(fltLine)
        # datMat[:,0] 表示矩阵的第一列的所有元素 dataMatrix[0,:] 表示矩阵的第一行的所有元素
        # datMat[:,1] 表示矩阵的第二列的所有元素 dataMatrix[1,:] 表示矩阵的第二行的所有元素
    return dataMat

# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 随机选择k个样本点作为初始聚类中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    # 初始化一个kxn的零矩阵，包含k(用户指定)个作为随机选择的聚类中心的样例
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        '''
        numpy.random.rand(d0,d1,…,dn)
            rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
            dn表格每个维度
            返回值为指定维度的array
        '''
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    # clusterAssment 仍旧为mx2的矩阵，用于存储每个样例的簇分配结果(第一列为簇索引值)和误差(某个样例点到其簇的距离平方值)
    clusterAssment = mat(zeros((m,2)))
    # centroid0 记录整个数据集所有样例的初始聚类中心
    '''
    numpy tolist() 将数组或者矩阵转换成列表
        特别说明 tolist()[0] 与 tolist() 的区别
            tolist()
                a1 = [[1,2,3],[4,5,6]] # a1 为列表
                a3 = mat(a1) # matrix([[1, 2, 3],[4, 5, 6]])
                a5 = a3.tolist() # [[1, 2, 3], [4, 5, 6]]
                a1 == a5 # Ture
            tolist()[0]             
                a1 =[1,2,3,4,5,6] # a1 为列表
                a3 = mat(a1) # matrix([[1, 2, 3, 4, 5, 6]])
                a4 = a3.tolist() # [[1, 2, 3, 4, 5, 6]]
                a1 == a4 # False
                a8 = a3.tolist()[0] # [1, 2, 3, 4, 5, 6]
                a1 == a8 # Ture
    '''
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # centList 作为一个列表来保存所有的聚类中心
    centList =[centroid0] #create a list with one centroid
    print("initial cenList: ")
    print(centList)
    # for循环的作用：计算每一个样例与初始聚类中心的欧式距离的平方值，并存入clusterAssment的第二列(第一列的索引值仍全部为0)
    #   额外说明：除Ada, DT, K-means, KNN, LR, NB, SMO 以外的所有算法，矩阵一律从第0行，第0列开始算起
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    print("initial clusterAssment: ")
    print(clusterAssment)
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            print("i="+str(i))
            # ptsInCurrCluster 得到当前第i簇的所有样例(mxn矩阵: m为样例数，n为某个样例包含的属性值的个数)
            # nonzero() 在此处的用法下面链接中的文章讲得非常清晰通透
            #   https://blog.csdn.net/ruibin_cao/article/details/83242489?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3
            # 总结nonzero() 在此处的用法: nonzero(clusterAssment[:, 0].A == i)[0]
            #   clusterAssment[:, 0].A == i 是一个判断表达式，即当clusterAssment矩阵第1列的每个值等于当前第i簇时，为True
            #   nonzero(clusterAssment[:, 0].A == i) 当nonzero()的()中是一个判断表达式时，如果为True，则返回元素的索引值
            #   nonzero(clusterAssment[:, 0].A == i)[0] [0]表示索引值的行坐标
            #   举例说明
            #       第一轮for循环，i = 0 时，clusterAssment[:, 0]全部为0，所以为True，所以应该返回clusterAssment中的所有元素的索引值，加上[0],则返回元素的行坐标，输出为：
            # [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59]
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 使用KMeans把当前第i簇划分成为两个簇
            #   centroidMat为这两个簇的聚类中心
            #   splitClustAs为一个mxn矩阵(m为第i簇中所有的样例总数；n为2，第一列为样例所属聚类中心的索引值，第二列为误差值)
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            print("the " + str(i) + " cluster is splited 2 clusters.")
            print(centroidMat)
            print(splitClustAss)
            # sseSplit 为i簇所有样例的误差总值(或可称为第i簇的误差值)
            sseSplit = sum(splitClustAss[:,1])
            # seeNotSplit 剩余数据集的误差总和，即除第i簇之外的样例的误差总和
            # clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1] 的举例分析
            #   第一轮for循环，当i=0时，nonzero(clusterAssment[:,0].A!=i)[0] 为 []
            #   clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1] 等价于 clusterAssment[[],1]
            print("indexes of samples not in " + str(i) + " cluster.")
            print(nonzero(clusterAssment[:,0].A!=i)[0])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            # 如果这次划分的总误差值小于最小误差值，则本次划分保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                print("i=" + str(i) + ", total SEE of the split  is less than lowest SEE, the split is valid!")
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 使用两个数组过滤器来更新簇分配结果
        print("hhhhhhhhhhhh")
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        # 举例分析(第一轮for循环，i=0) clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]
        #   clusterAssment[:, 0].A == bestCentToSplit clusterAssment 的第一列全为0，bestCentToSplit = i = 0
        #   所以，clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] 包含 clusterAssment中的所有样例
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss #reassign new clusters, and SSE
        print("centList has been updated: ")
        print(centList)
        print("clusterAssment has been updated: ")
        print(clusterAssment)
    return mat(centList), clusterAssment

datMat = mat(loadDataSet('testSet2.txt'))
centList, myNewAssments = biKmeans(datMat,3)
print(myNewAssments)
