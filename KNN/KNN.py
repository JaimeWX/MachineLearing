from numpy import *
import operator
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# KNN算法
def classify0(inX, dataSet, labels, k): # intX表示将要被分类的输入向量，例如[3,2]; k表示最近邻居的数目
    dataSetSize = dataSet.shape[0]
    # tile()的作用：把[3,2]横向复制成为[[3,2] [3,2] [3,2] [3,2]]
    # (算法核心) 以下4行代码 一次计算出了inX与训练数据集中所有点的欧式距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort()的作用：将列表中的元素从小到大排序，返回的是元素的索引。例如 [0.3,0.1,0.8,1.0] 返回[1,0,2,3]
    # 把训练数据集重新排序根据与inX的欧式距离(从小到大) !!sortedDistIndicies中为原训练数据集的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        # dic.get(key,default=None) 返回指定键的值 !!如果指定键的值不存在时，返回该默认值。
        classCount[voteIlable] = classCount.get(voteIlable,0) + 1
    # operator.itemgetter()的作用：举例说明
        # 若 a = [1, 2, 3] 则 itemgetter(1) 为 2 则 itemgetter(2,1) 为 [3,2]
        # 若 b = [[1,2,3],[4,5,6],[7,8,9]] 则 itemgetter(1) 为 [4,5,6] 则 itemgetter(2,1) 为 [[7,8,9],[4,5,6]]
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 从文本文件中解析数据(得到datMAt,labels)
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 归一化特征值：将任意取值范围的特征值转化为0到1的值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m为normMat中的样例总数
    m = normMat.shape[0]
    # 测试数据集的数量为normMat总数的10%
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #print(normMat[i,:])
        #print(normMat[numTestVecs:m,:])
        # 详细解释for循环
        #   normMat[i,:] 表示normMat的第i行
        #   normMat[numTestVecs:m,:] 表示normMat的第numTestVecs到第m行
        #   测试集为norMat的前100行，训练集为norMat的后900行
        #   for循环把测试集的每个样例都用knn算法进行测试，举例说明
        #       classify0(normMat[0, :], normMat[100:1000, :], datingLabels[100:1000], 3)
        #       表示把[0.44832535 0.39805139 0.56233353]与训练集中的900个样例求欧式距离......
        #       (解释得够清楚了，在这个方法中关键在于找出训练集是什么，测试集是什么）
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

# 简化版 简化内容：inArr中一个示例的三个属性值应该是用户输入的
def classifyPerson():
    #这里翻译应该是是'毫无魅力'，'一般魅力'，'特别有魅力'
    resultList = ['not at all', 'in small doses', 'in large doses']
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([10000,10,0.5])
    cR = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print(resultList[cR-1])


#test1
#group,labels = createDataSet()
#classify0([3,2],group,labels,3)

# test2
# datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

'''
test3
# 使用Matplotlib创建散点图
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
# datingDataMat[:,0], datingDataMat[:,1] 为datMat的第一维与第二维，分别表示散点图的x轴与y轴
# 15.0*array(datingLabels), 15.0*array(datingLabels) 利用颜色及尺寸标识了数据点的属性类别
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
'''
#test4
#normMat,ranges,minVals = autoNorm(datingDataMat)

#test5
#datingClassTest()

#test6
#classifyPerson()

