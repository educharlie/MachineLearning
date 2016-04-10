#Goal: Select whether the Newspaper feature should be included in the linear regression model on the advertising dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

# 10-fold cross-validation with all three features
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
#print scores
# fix the sign of MSE scores
mse_scores = -scores
# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
# calculate the average RMSE
print rmse_scores.mean()

# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
print np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean()