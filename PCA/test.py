# 取矩阵的前x列或前x行
import numpy as np

A = np.mat('80,100,40,20; 100,170,140,30; 40,140,200,50')
print(A)
print(A[:2,:])
print(A[:,:2])