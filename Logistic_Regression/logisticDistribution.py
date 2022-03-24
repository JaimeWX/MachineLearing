# logistic分布的分布函数F(x)与密度函数f(x)

from sympy import *

x,mu,gamma = symbols('x mu gamma')
F_x = 1 / (1+exp(-(x-mu))/gamma)
f_x = diff(F_x,x)
print(f_x)
f_x = exp(mu - x)/(gamma*(1 + exp(mu - x)/gamma)**2)