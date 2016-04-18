from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors = 5)
k_range = range(1, 31)

weight_options = ['uniform', 'distance'] #a new varaible
param_grid = dict(n_neighbors = k_range, weights = weight_options)

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(X,y)
grid.grid_scores_
#print grid.best_score_
#print grid.best_params_


#Using the best parameters to make predictions
#train your model using all data and the best known parameters
knn = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')
knn.fit(X, y)
#print knn.predict([3, 5, 4, 2])
#shorcut: same result
#print grid.predict([3, 5, 4, 2])


#Reducing computational expense using RandomizedSearchCV
from sklearn.grid_search import RandomizedSearchCV
#specify "parameter distribution" rather than a "parameter grid"
param_dist = dict(n_neighbors = k_range, weights = weight_options)
#n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10, random_state = 5)
rand.fit(X, y)
rand.grid_scores_
#print rand.best_score_
#print rand.best_params_

#run RandomizedSearchCV 20 times (with n_iter = 10) and record the best score
best_scores = []
for _ in range(20):
	rand = RandomizedSearchCV(knn, param_dist, cv = 10, scoring = 'accuracy', n_iter = 10)
	rand.fit(X,y)
	best_scores.append(round(rand.best_score_, 3))
print best_scores
