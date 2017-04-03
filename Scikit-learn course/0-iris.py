#import load_iris function from dataset module
from sklearn.datasets import load_iris
#save "bunch" object containing iris dataset and its distribution
iris = load_iris()
type(iris)
#print iris data
print iris.data
#print the names of the four features
print iris.feature_names
#print integers representing the species of the observation
print iris.target
#print the encoding scheme for species: 0=setosa, 1=versicolor, 2=virginica
print iris.target_names
#check the types of the features and response
print type(iris.data)
print type(iris.target)
#check the shape of the features (first dimension=number of the observations, second dimension=number of features)
print iris.data.shape
#check the shape of the response (single dimension matching the number of observations)
print iris.target.shape
X = iris.data #matrix
y = iris.target #vector