from numpy import *

# Q1: <统计学习方法>中的算法描述中说初始化时随机选择k个样本点作为初始聚类中心，但是 randCent(dataSet, k) 并没有选择具体的样本点，而是在
#     以datMat中的最大值与最小值为范围随机选取了一个"点"
#     eg. 初始聚类中心为[-1.98654506 -1.80136702],但该"点"并不是daMat中的某个向量

'''
算法总结
    s1 初始化k(用户指定)个聚类中心
    s2 分配所有样例点到这k个聚类中心(根据点与中心的最小欧式距离)
    s3 更新这k个聚类中心的值(根据聚类中心中所含的所有样例点的均值)
    s4 不断重返s2,s3,直到每个聚类中心中的样例不再改变(均值不变)
    (K-Means相对来说比较简单，并没有打印runResults并对其进行详细分析)
'''

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
    # m 样例总数
    m = shape(dataSet)[0]
    # clusterAssment 一个矩阵用来存储每一个点的簇分配结果
    #   第一列记录簇索引值
    #   第二列存储误差
    clusterAssment = mat(zeros((m,2)))
    # 随机生成k个初始聚类中心
    #   注：形参createCent 传递了一个 def，randCent(dataSet, k)
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        # clusterChanged 作用是保证直到所有样例的簇分配结果不再改变时退出下面的双重for循环
        clusterChanged = False
        # 双重for循环解释
        #   外循环遍历所有的样例，内循环遍历所有的聚类中心
        #   计算每一个样例与每一个聚类中心的欧式距离，把每一个样例分配到欧式距离最小的簇中
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 再次强调，第j行，第i行
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 第二列保存着每个点的误差，即某个样例点到其簇的距离平方值，这个误差值用于评价聚类的质量
            clusterAssment[i,:] = minIndex,minDist**2
        #print(centroids)
        # for循环解释：遍历所有聚类中心并更新它们的取值
        #   找到每一个簇中的所有样例点，计算这些样例点的均值，以此均值更新该簇的聚类中心的值
        for cent in range(k):
            # 找到每一个簇中的所有样例点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]#get all the point in this cluster # 再次强调：.A 就是把矩阵转换成为数组
            '''
            mean()函数功能：求取均值
                经常操作的参数为axis，以m * n矩阵举例：
                    axis 不设置值，对 m*n 个数求均值，返回一个实数
                    axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
                    axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
            '''
            # 计算这些样例点的均值，以此均值更新该簇的聚类中心的值
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment


datMat = mat(loadDataSet('testSet.txt'))
myCentroids,clustAssing = kMeans(datMat,4)
