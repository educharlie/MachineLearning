from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()
X = iris.data
y = iris.target
#change random state give me different prediction (high varaince estimate problem)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X)
print metrics.accuracy_score(y, y_pred)

#K fols Cross Validation
#simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds = 5, shuffle = False)
#print the contents of each training and testing set
print '{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations')
for iteration, data in enumerate(kf, start = 1):
	print '{:^9} {} {:^25}'.format(iteration, data[0], data[1])

#Select the best tunig parameters (aka hyperparameters) for KNN on the iris dataset
from sklearn.cross_validation import cross_val_score
#10-folds cross validation with K=5 for KNN
knn = KNeighborsClassifier(n_neighbors = 5) 
scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy') #scoring = classification accuracy as evaluation metric
print scores
#use average accuracy as a estiamte of out of sample accuracy
print scores.mean()

#search for the optimal value of K for KNN
k_range = range(1, 31)
k_scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors = k)
	scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
	k_scores.append(scores.mean())
print k_scores
import matplotlib.pyplot as plt
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#plt.show()

#Compare the best KNN model with logistic regression on the iris data set
# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=20)
print cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean()

