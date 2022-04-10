import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

### 感知机算法
# 定义sign符号函数
def sign(x,w,b):
    # dot()返回的是两个数组的点积
    #   如果处理的是一维数组，则得到的是两数组的內积
    #   如果是二维数组（矩阵）之间的运算，则得到的是矩阵积
    return np.dot(x,w)+b

# 定义参数初始化函数
def initialize_parameters(dim):
    '''
    输入：
        dim：输入数据维度
    输出：
        w：初始化后的权重系数
        b：初始化后的偏置参数
    '''
    w = np.zeros(dim,dtype=np.float32)
    b = 0.0
    return w,b

# 定义感知机训练函数(核心算法)
def train(X_train, y_train, learning_rate):
    # 参数初始化
    # w = [0. 0.] b = 0.0
    w, b = initialize_parameters(X_train.shape[1]) # 维度即对应列数！！！
    # 初始化误分类
    is_wrong = False
    while not is_wrong:
        wrong_count = 0
        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]
            # 如果存在误分类点
            # 更新参数
            # 直到没有误分类点
            if y * sign(X, w, b) <= 0:
                w = w + learning_rate * np.dot(y, X)
                b = b + learning_rate * y
                wrong_count += 1
        if wrong_count == 0:
            is_wrong = True
            print('There is no missclassification!')
        # 保存更新后的参数
        params = {'w': w,'b': b}
    return params

### 处理训练数据集
iris = load_iris()
# 转化为pandas数据框
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 数据标签
df['label'] = iris.target
# 变量重命名
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# 取前100行数据作为训练数据集(我认为其原因是感知机为二分类线性分类模型，此数据集的前一百行正好对应2种类别)
data = np.array(df.iloc[:100, [0, 1, -1]])
# 定义训练输入与输出
# X 为原始数据集的前2列，y 把原始训练集的Iris-setosa转换成了-1；Iris-versicolor转换成了1
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])

### 求解参数w，b
params = train(X,y,0.01)
print(params)

### 画图
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='red', label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='green', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
x_points = np.linspace(4, 7, 10)
y_hat = -(params['w'][0]*x_points + params['b'])/params['w'][1]
plt.plot(x_points, y_hat)
plt.show()





