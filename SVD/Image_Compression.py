# SVD图像压缩的原理：图像清晰度随着奇异值数量增多而提高。当奇异值k的数量不断增加，恢复后的图像就会无限逼近真实图像。
'''
SVD图像压缩(恢复)算法运行过程分析
    先对输入图片((1280, 960, 3))分别进行RGB三个通道的奇异值分解，求出RGB中每一个的U(1280,1280)，S(1280,960)，Vt(960,960)
    然后恢复图片
        以 k=1(2个奇异值) R = restore(u_r, s_r, v_r, 2) 为例
            当k=0时
                u0(1280,1) = u[:, 0](u的第一列).reshape(m,1) # m=1280
                v0(1,960) = v[0](v的第一行).reshape(1,n) # n=960
                s0(是一个数，因为np.linalg.svd()不会有S矩阵，只是把奇异值保存下来) = s[0]
            a0 = u0 * s0 * v0 (输入图片仅使用一个奇异值时恢复得到的R)
            当k=1时 同理得到a0+a1 (输入图片使用两个奇异值时恢复得到的R)
            同理得到G与B
'''

import numpy as np
import os
# PIL(Python Imaging Library)为python解释器提供了图像编辑函数。
from PIL import Image
from tqdm import tqdm

# step1 读入待压缩图像
# PIL.Image.open()打开并标识给定的图像文件。
#   这是一个懒惰的操作；此函数可识别文件，但文件保持打开状态，直到尝试处理数据(或调用load()方法)，才会从文件中读取实际图像数据。
# PIL读出来的图片size应该是(width,height)，但是转成numpy矩阵后，变成了(height, width, channels)
#   A.shape为1280, 960, 3)
#       A.shape[:, :, 0]为(1280, 960)
#       A.shape[:, :, 1]为(1280, 960)
#       A.shape[:, :, 2]为(1280, 960)
A = np.array(Image.open("./louwill.jpg", 'r'))

# step2 对RGB图像进行奇异值分解
# RGB图片也可以看作是三层二维数组的叠加，每一层二维数组都是一个通道。
u_r, s_r, v_r = np.linalg.svd(A[:, :, 0])
u_g, s_g, v_g = np.linalg.svd(A[:, :, 1])
u_b, s_b, v_b = np.linalg.svd(A[:, :, 2])


# 定义恢复函数，由分解后的矩阵恢复到原矩阵
def restore(u, s, v, K):
    '''
    u:左奇异矩阵
    v:右奇异矩阵
    s:奇异值矩阵
    K:奇异值个数
    '''
    m, n = len(u), len(v[0])
    print(m,n)
    a = np.zeros((m, n))
    for k in range(K):
        print("k="+str(k))
        uk = u[:, k].reshape(m, 1)
        print(u.shape)
        print(uk.shape)
        vk = v[k].reshape(1, n)
        print(vk.shape)
        # 前k个奇异值的加总
        print(s[k])
        a += s[k] * np.dot(uk, vk)
        print(a.shape)
    a = a.clip(0, 255)
    # uint8是8位无符号整型，uint16是16位无符号整型
    return np.rint(a).astype('uint8')

# 使用前50个奇异值
K = 50
output_path = r'./svd_pic'
# tqdm是 python的一个关于进度条的扩展包，在深度学习进程中可以将训练过程用进度条的形式展现出来，会让训练界面更加的美观。
for k in tqdm(range(1, K+1)):
    R = restore(u_r, s_r, v_r, k)
    G = restore(u_g, s_g, v_g, k)
    B = restore(u_b, s_b, v_b, k)
    I = np.stack((R, G, B), axis=2) # stack()的作用是使3个(1280 960)叠加到一起变成(1280, 960,3)
    Image.fromarray(I).save('%s\\svd_%d.jpg' % (output_path, k))