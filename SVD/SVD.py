# 奇异值分解
#   <线性代数及其应用> 7.4 eg3

import numpy as np

A = np.array([[4,11,14],[8,7,-2]])
U, s, Vt = np.linalg.svd(A, full_matrices=True)

'''
某次实验结果：s = [18.97366596  9.48683298]
    s并不是所期待的2x3矩阵，原因是linalg.svd对奇异值矩阵做了简化，只给出了奇异值向量，省略了奇异值中为0的部分
'''
# print(U.shape, s.shape, Vt.shape)
# print(U)
# print(s)
# print(Vt)

# 得到正常的S
#   np.diag_indices_from() 要求矩阵必须是方阵
S = np.eye(A.shape[0],A.shape[1])
for i in range(len(S)):
    S[i][i] = s[i]
# 验证
print(np.allclose(A, np.dot(np.dot(U,S),Vt)))