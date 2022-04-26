import numpy as np

class PCA():
    # 计算协方差矩阵
    def calc_cov(self, X):
        m = X.shape[0]
        # 数据标准化: X按照列均值为0，方差为1进行标准化处理
        X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)
        # X的协方差矩阵为 1/m * X * X.T
        return 1 / m * np.matmul(X.T, X)

    def pca(self, X, n_components):
        # 计算协方差矩阵
        cov_matrix = self.calc_cov(X)
        # 计算协方差矩阵的特征值和对应特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # 对特征值排序
        idx = eigenvalues.argsort()[::-1]  # 此处按降序排列。如果没有[::-1]则按升序排列
        # 取最大的前n_component组(可以理解为将原始数据降维到n_component个主成分)
        eigenvectors = eigenvectors[:, idx]  # 将特征向量按照对应特征值大小排列成矩阵
        eigenvectors = eigenvectors[:, :n_components]  # [:, :n_components] 取X的前n_components列 [:_components,:n] 取X的前n_components行
        # Y=PX转换 （P为4x3 X为150x4 Y为150x3 可见降了一维）
        return np.matmul(X, eigenvectors)
