#Allows you to define a grid of parameters that will be searched using K-fold cross validation
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors = 5) 

#define the parameter values that should be searched 
k_range = range(1, 31)
#create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors = k_range)
#instantiate the grid: You can set n_jobs = -1 to run computations in parallel (if supported by your computer and OS)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
#fit the grid with data
grid.fit(X,y)
#view of the complete results (list of manual tuples)
grid.grid_scores_

#examinate the first tuple 
'''print grid.grid_scores_[0].parameters
print grid.grid_scores_[0].cv_validation_scores
print grid.grid_scores_[0].mean_validation_score'''

#create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
#print grid_mean_scores

#plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of the K for KNN')
plt.ylabel('Cross Validation Accuracy')
#plt.show()

#examinate the best model
print grid.best_score_
print grid.best_params_
print grid.best_estimator_
