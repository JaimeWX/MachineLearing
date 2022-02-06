from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 上述6个词条集合的类别由人工标为[0, 1, 0, 1, 0, 1]，这些标注信息用于训练程序以便自动检测侮辱性留言
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

# 把输入的多个词条集合中的所有词条放入一个集合vocabSet中，并且没有重复的词条(词集模型(set-of-words model),每个词只能出现一次）
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个集合的并集
    return list(vocabSet)

# 判断一个词条集合中的每一个词条是否在vocabSet中，如果在则为1，不在则为0。如果不都不在，则 vocabSet = [0...0]
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print
        "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    # 在此例中，trainMatrix是由6个向量，每个向量由32个元素组成的矩阵.(32个元素是因为6个词条集合中共有32个不重复的词)
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 求先验概率P(c = abusive)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 总体上在求条件概率，使用了极大似然估计后验概率
    # numpy.ones 返回一个指定形状和数据类型的新数组，并且数组中的值都为1
    # 使用 ones 和 p0Denom = 2.0; p1Denom = 2.0 的原因是做了拉普拉斯平滑(lambda = 1，S_j = 2)，解决连乘造成估计的概率值为0的问题
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i] )
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对数似然，防止下溢
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#testingNB()









