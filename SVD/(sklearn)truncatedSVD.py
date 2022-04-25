from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

# 创建100x100的稀疏矩阵(其中有100个取值在0-1范围内的非零元素)
X = sparse_random(100,100,density=0.01,format='csr',random_state=42)
# 基于截断SVD算法对X进行降维，降维的维度为5，即输出前5个奇异值
svd = TruncatedSVD(n_components=5,n_iter=7,random_state=42)
svd.fit(X)
print(svd.singular_values_)