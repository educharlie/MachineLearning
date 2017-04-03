#diferent classification model

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

logReg = LogisticRegression()

logReg.fit(X,y)

X_new = [[3, 5, 4, 2],[5, 4, 3, 2]]
result = logReg.predict(X_new)
print result