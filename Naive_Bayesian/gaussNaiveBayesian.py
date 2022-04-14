from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
# print(X)
# print(y)
# random_state：是随机数的种子。
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
# 比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# print(len(X_train))
# print(len(X_test))
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy of GaussianNB in iris data test:",
      accuracy_score(y_test, y_pred))