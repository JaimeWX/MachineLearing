import numpy as np
'''
算法总结
    对应P210eg10.3 (δ,ψ从1开始)
        此题递推只有两轮，即t=2，t=3
            当 t = 2时
                先求出所有的δ1(j)*aji放入tmp矩阵
                根据tmp结合max即可求得ψ2(i)
                根据tmp结合max再乘b1o(2)即可求得δ2(i)
            当 t = 3时,同理
        终止 ...
        回朔最优路径
            前面已经求出了所有ψ1(1)-ψ3(3)并放入了varphi矩阵
            只需要根据i_t^* = ψ_t+1(i_t+1^*)求出即可
算法修改
    使用len(delta)表示状态集合Q的个数
'''
### 序列标注问题和维特比算法
def viterbi_decode(O,A,B,pi):
    '''
    输入：
    O：观测序列, λ = (A,B,π)
    输出：
    path：最优隐状态路径I*
    '''
    # 序列长度和初始观测
    T, o = len(O), O[0]
    # 初始化delta变量
    delta = pi * B[:, o]   # δ1(1) = π1 * b1(o1) δ1(2) = π2 * b2(o1) δ1(3) = π3 * b3(o1)
    # 初始化varphi变量
    # 用len(delta)来表示状态集合的个数
    # 初始化了一个全为0的len(O) x len(delta)的二维矩阵(ψ1(i)必为0，此处先使ψ2(i),...,ψT(i)也为0）
    varphi = np.zeros((T, len(delta)), dtype=int)
    # I*=(i1*,i2*,...,iT*)
    path = [0] * T
    # 递推
    for i in range(1, T):
        delta = delta.reshape(-1, 1)   # reshape()在此处的作用等于转置
        # tmp包含a11 - a33包含所有的需要：
        #   求解 δ2(1) 需要δ1(1)*a11,δ1(2)*a21,δ1(3)*a31中的最大值;
        #   求解 δ2(2) 需要δ1(1)*a12,δ1(2)*a22,δ1(3)*a32中的最大值;
        #   求解 δ2(3) 需要δ1(1)*a13,δ1(2)*a23,δ1(3)*a33中的最大值;
        tmp = delta * A
        print(tmp)
        # np.argmax()详见argmax.py
        # 当i=1时，即求解δ2(1),δ2(2),δ2(3)时
        #   ψ2(1) = δ1(1)*a11,δ1(2)*a21,δ1(3)*a31中的最大值的索引；ψ2(2)，ψ2(3)同理
        varphi[i, :] = np.argmax(tmp, axis=0)
        print(varphi[i, :])
        # 当i=1时，即求解ψ2(1),ψ2(2),ψ2(3)时
        #   δ2(1) = δ1(3)*a31*b1(o2)
        #   δ2(2) = δ1(3)*a32*b1(o2)
        #   δ2(3) = δ1(3)*a33*b1(o2)
        delta = np.max(tmp, axis=0) * B[:, O[i]]
        print(delta)
    # 终止
    path[-1] = np.argmax(delta)
    # 回溯最优路径
    #
    for i in range(T-1, 0, -1):   # 这里使步长为负数，当T=3时，i先为2再为1
        print(varphi)
        path[i-1] = varphi[i, path[i]]
        print(path[i-1])
    return path

# 给定观测
O = [0,1,0]
# 初始状态概率分布
pi = np.array([0.2,0.4,0.4])
# 状态转移概率矩阵
A = np.array([
    [0.5,0.2,0.3],
    [0.3,0.5,0.2],
    [0.2,0.3,0.5]])
# 观测概率矩阵
B = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]])

print(viterbi_decode(O,A,B,pi))