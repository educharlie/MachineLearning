#Evaluation procedure 2: Train and test split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
iris = load_iris()
X = iris.data
y = iris.target

#Step 1: Split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4) #0,4 = 40%, random_state: no random in each run
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape 
#Step 2: Train the model on the training set
logReg = LogisticRegression()
logReg.fit(X_train, y_train)
#Step 3: Make predictions on the testing set
y_pred = logReg.predict(X_test)
#Compare responses
#print metrics.accuracy_score(y_test, y_pred) 

#K = 5
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
#print metrics.accuracy_score(y_test, y_pred)

#K = 1
knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
#print metrics.accuracy_score(y_test, y_pred) #best option!

#Which is the best k value?
#try K=1 to K=25 and record testing accuracy
k_range = range(1, 26)
scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors = k) 
	knn.fit(X_train,y_train)
	y_pred = knn.predict(X_test)
	scores.append(metrics.accuracy_score(y_test, y_pred))
#print scores

#plotting values
from matplotlib import pyplot as plt
plt.plot(k_range, scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Testing Accuracy")
#plt.show()

#once you decide what is the best K, you train your data with the all data set
knn = KNeighborsClassifier(n_neighbors = 11) 
knn.fit(X,y)
y_pred = knn.predict([3, 5, 4, 2])
print y_pred

