# 导入sklearn数据集
from sklearn import datasets
import matplotlib.pyplot as plt
from numpy_pca import PCA

iris = datasets.load_iris()
X = iris.data  # 150x4
y = iris.target  # 150

# 将数据降维到3个主成分
X_trans = PCA().pca(X, 3)

# 颜色列表
colors = ['navy', 'turquoise', 'darkorange']
# 绘制不同类别
for c, i, target_name in zip(colors, [0,1,2], iris.target_names):
    plt.scatter(X_trans[y == i, 0], X_trans[y == i, 1],
            color=c, lw=2, label=target_name)
# 添加图例
plt.legend()
plt.show();
