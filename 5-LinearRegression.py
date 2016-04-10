
#Pandas libray: Python popular library for data exploration, manipulation and analysis
import pandas as pd
#read a cvs file 
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0) #index_col: set the col which would be the index
#display the first 5 rows
#print data.head()
#display the last 5 rows
#print data.tail()

#Seaborn: Python library for statistical visualization built on top of Matplotlib
import seaborn as sns
#sns.pairplot(data, x_vars = ['TV','Radio','Newspaper'], y_vars = ['Sales'], size = 7, aspect = 0.7)
sns.pairplot(data, x_vars = ['TV','Radio','Newspaper'], y_vars = ['Sales'], size = 7, aspect = 0.7, kind = 'reg') #with linear regression
import matplotlib.pyplot as plt
#plt.show()

#linear Regression
#Preparing X
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
#print X.head()
#Preparing y
y = data['Sales']
#print y.head()

#Splitting X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1) #default split is 75% for training and 25% for testing

#Linear Regression
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X_train, y_train)
#print the intercerpts and coeficients
print linReg.intercept_ #output: 2.87
print linReg.coef_ #output [0.04  0.17  0.003] function: y=2.88 + 0.04 x TV + 0.17 x Radio + 0.003 x Newspaper
#pair the feature names with the coefficients
print zip(feature_cols, linReg.coef_)
#Making predictions
y_pred = linReg.predict(X_test)

#Model evaluation metrics for regression
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
from sklearn import metrics
#print metrics.mean_absolute_error(true, pred) #MAE
#print metrics.mean_squared_error(true, pred) #MSE
import numpy as np
#print np.sqrt(metrics.mean_squared_error(true, pred)) #RMSE

#Choose RMSE for our prediction
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#As we've seen in the plot, newspaper doesn't have a linear co-relation, let's remove it
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
y = data.Sales
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

