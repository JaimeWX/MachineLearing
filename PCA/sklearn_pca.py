# 导入sklearn降维模块
from sklearn import decomposition # 基于奇异值分解法实现
from sklearn import datasets
import matplotlib.pyplot as plt

# 创建pca模型实例，主成分个数为3个
pca = decomposition.PCA(n_components=3)

# 导入数据集
iris = datasets.load_iris()
X = iris.data  # 150x4
y = iris.target  # 150

# 模型拟合
pca.fit(X)
# 拟合模型并将模型应用于数据X
X_trans = pca.transform(X)

# 颜色列表
colors = ['navy', 'turquoise', 'darkorange']
# 绘制不同类别
for c, i, target_name in zip(colors, [0,1,2], iris.target_names):
    plt.scatter(X_trans[y == i, 0], X_trans[y == i, 1],
            color=c, lw=2, label=target_name)
# 添加图例
plt.legend()
plt.show();