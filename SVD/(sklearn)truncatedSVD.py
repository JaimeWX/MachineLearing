# 截断奇异值分解

from sklearn.decomposition import TruncatedSVD
# 导入Scipy生成稀疏数据模块
from scipy.sparse import random as sparse_random

# 创建稀疏数据X
X = sparse_random(10,10,density=0.01,format='csr',random_state=42)
print(X)