#import load_iris function from dataset module
from sklearn.datasets import load_iris

#save "bunch" object containing iris dataset and its attributes
iris = load_iris()

#store features matrix in "X"
X = iris.data
#store response vector in "y"
y = iris.target

#print the shapes of X and y
#print X.shape #verify if the features were loaded correctly
#print y.shape #verify if the response was loaded correctly

#4-steps modeling pattern

#Step 1: Import the class you plan to use
from sklearn.neighbors import KNeighborsClassifier

#Step 2: "Instantiate" the estimator: "Estimator" is scikitlearn's term for model, Instantiate means "make a instance of"
knn = KNeighborsClassifier(n_neighbors = 1) #k=1, the rest of the parameters are default
#print the default vaules
#print knn

#Step 3: Fit the model with data aka "model trainig" 
#Model is learning the relationship between X and y 
knn.fit(X,y)

#Step 4: Predict the response for a new observation
#New observation are called "out-of-sample" data
#Uses the information it learned during the model training process
result = knn.predict([3, 5, 4, 2])
print result #2 = Virginica
#Return an NumPy array

#Can predict for multiple observations at once
X_new = [[3, 5, 4, 2],[5, 4, 3, 2]]
otherResult = knn.predict(X_new)
print otherResult

#k=5
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)
otherResult = knn.predict(X_new)
print otherResult