'''
算法总结
    ### input: λ = (A,B,π) 和 观测序列的长度T
    ### ouput: 观测序列O = (o1,o2,...,oT).

    在该代码中体现却没有在书上的伪代码中体现出来的东西(r -= p)
            举例说明:当按照初始状态分布(0.25, 0.25, 0.25, 0.25)产生初始状态i0时
                使用了np.random.rand()产生了一个服从“0~1”均匀分布的随机样本值r(随机样本取值范围是[0,1)，不包括1)，假设r=0.7
                ix = 0 , p = 0.25
                    r > p , r = r-p = 0.45
                ix = 1 , p = 0.25
                    r > p , r = r-p = 0.2
                ix = 2 , p = 0.25
                    r < p, 所以初始状态i0就选第三个盒子！！！
            包括o0,i1,o1,...,o4所采用的都是与这个相同的逻辑。
'''
import numpy as np
# 定义HMM模型
class HMM:
    def __init__(self, N, M, pi=None, A=None, B=None):
        # 可能的状态数
        self.N = N
        # 可能的观测数
        self.M = M
        # 初始状态概率向量
        self.pi = pi
        # 状态转移概率矩阵
        self.A = A
        # 观测概率矩阵
        self.B = B

    # 根据给定的概率分布随机返回数据
    def rdistribution(self, dist):
        # np.random.rand(d0,d1,d2……dn) 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
        r = np.random.rand()
        print("r= " + str(r))
        '''
        enumerate() 在for循环中的使用示例
        >>> seq = ['one', 'two', 'three']
        >>> for i, element in enumerate(seq):
                print i, element
        >>> output:
                0 one
                1 two
                2 three
        '''
        for ix, p in enumerate(dist):
            print("ix= "+str(ix))
            print("p="+str(p))
            if r < p:
                return ix
            r -= p

    # 生成HMM观测序列
    def generate(self, T):
        # 根据初始概率分布生成第一个状态
        i = self.rdistribution(self.pi)
        print("aaa")
        print("i0="+str(i))
        # 生成第一个观测数据
        o = self.rdistribution(self.B[i])
        print("bbb")
        print("o0="+str(o))
        observed_data = [o]
        # 遍历生成剩下的状态和观测数据
        t = 1
        for _ in range(T-1):
            i = self.rdistribution(self.A[i])
            print("i"+str(t)+"="+str(i))
            o = self.rdistribution(self.B[i])
            print("o" + str(t) + "=" + str(o))
            observed_data.append(o)
        return observed_data


# 初始状态概率分布
pi = np.array([0.25, 0.25, 0.25, 0.25])
# 状态转移概率矩阵
A = np.array([
    [0,  1,  0, 0],
    [0.4, 0, 0.6, 0],
    [0, 0.4, 0, 0.6],
    [0, 0, 0.5, 0.5]])
# 观测概率矩阵
B = np.array([
    [0.5, 0.5],
    [0.6, 0.4],
    [0.2, 0.8],
    [0.3, 0.7]])
# 可能的状态数和观测数
N = 4
M = 2
# 创建HMM实例
hmm = HMM(N, M, pi, A, B)
# 生成观测序列(T = 5)
print(hmm.generate(5))