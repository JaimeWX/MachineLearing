from math import log
import operator

# 直接使用信息熵公式计算为 -(0.4*log(0.4,2)+0.6*log(0.6,2))
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 训练集合中的样例总数为5
    labelCounts = {}
    for featVec in dataSet: # {'yes': 2, 'no': 3} for循环的作用是创建一个字典，键为训练集合的标记，值为不同标记的示例数量
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts: # for循环的作用是使用信息熵公式计算训练集合的信息熵Ent(D)
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 这个函数的功能是根据某一个属性划分训练集。例如，色泽只有两个属性值，青绿和乌黑，青绿的为一类，乌黑的为另一类。
def splitDataSet(dataSet,axis,value ):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 一个样例减去标记即为一个示例的所有属性
    baseEntropy = calcShannonEnt(dataSet)  # Ent(D)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):  # 若以myDat为训练集，numFeatures = 2，i = 0，1
        # 若以myDat为训练集，把第一个属性的所有属性值放入一个列表中，再把第二个属性的所有属性值放入一个列表中, 再把这两个列表放入featList
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)       # 若以myDat为训练集，set为{0,1}
        newEntropy = 0.0
        for value in uniqueVals:
            # 若以myDat为训练集，有2个属性，每个属性均有2个属性值，所以总共要求4个subDataSet
            subDataSet = splitDataSet(dataSet, i, value)
            # 下面所有代码用于计算Gain(D,'no surfacing'),Gain(D,'flippers'),并比较二者的大小(找出最优划分属性)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #['yes', 'yes', 'no', 'no', 'no']
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 基线条件1 stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # 基线条件2 stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
# 以上为ID3算法

# 构造好决策树之，使用决策树进行实际分类，即给定一个示例，判别它的类别。eg：[1,0] 的类别为 no
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断valueOfFeat是否为字典
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

# 使用模块pickle序列化对象，可在磁盘上保存对象，并在需要的时候读取出来
# 因为构造决策树会耗费很多决策时间，应在每次执行分类任务时调用已经构建好的决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

'''
# 2个属性，每个属性均有2个属性值；二分类
def craeteDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels=  ['no surfacing','flippers']
    return dataSet, labels
'''

'''
myDat,labels = craeteDataSet()
myTree = treePlotter.retrieveTree(0)
classify(myTree,labels,[1,0])
storeTree(myTree,'classifierStorage.txt')
print(grabTree('classifierStorage.txt'))
'''


