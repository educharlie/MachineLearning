#Evaluation procedure 1: Train and test on the entire set
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X,y)
y_pred = logReg.predict(X)
print y_pred

#1) Compute classifcation accuracy for the Logistic Regression model
from sklearn import metrics
print metrics.accuracy_score(y, y_pred) #called train accuracy

#2) Compute classifcation accuracy for the KNN k=5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X,y)
y_pred = knn.predict(X)
print metrics.accuracy_score(y, y_pred)

#3) Compute classifcation accuracy for the KNN k=1
knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(X,y)
y_pred = knn.predict(X)
print metrics.accuracy_score(y, y_pred) #best option!