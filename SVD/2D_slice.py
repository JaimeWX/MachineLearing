# 三维矩阵的切片
from PIL import Image
import numpy as np

A = np.array(Image.open("./louwill.jpg", 'r'))
print(A.shape)
print(A[:, :, 0].shape)
print(A[:, :, 1].shape)
print(A[:, :, 2].shape)
