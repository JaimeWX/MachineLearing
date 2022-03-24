from numpy import *
a = [[1,2,3],[4,5,6]]
A = mat(a)
print(A)
I = eye(shape(A)[0])
print(I)
'''
numTestPts = 30
for i in range(numTestPts):
    ws = exp(i - 10)
    print(ws)
'''
x = 2
ans = exp(x)
print(ans)

Amean = mean(A,0)
print(Amean)

ws = zeros((8,2))
print(ws)

for sign in [-3,3]:
    print(sign)