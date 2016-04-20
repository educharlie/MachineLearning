#Classication accuracy
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header = None, names = col_names)

#Can we predict the diabetes status of a patient given their health measurements?
# define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima.label

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)

#Classification accuracy: percentage of correct predictions
#Calculate accuracy
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)

'''Calculate Null accuracy: accuracy that could be achieved by always predicting the most frequent class
# examine the class distribution of the testing set (using a Pandas Series method)'''
y_test.value_counts()
# calculate the percentage of ones
y_test.mean()
# calculate the percentage of zeros
1 - y_test.mean()
# calculate null accuracy (for binary classification problems coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())
# calculate null accuracy (for multi-class classification problems)
y_test.value_counts().head(1) / len(y_test)

#Comparing the true and predicted response values
# print the first 25 true and predicted responses
print 'True:', y_test.values[0:25]
print 'Pred:', y_pred_class[0:25]


'''Confusion Matrix
#Table that describes the performance of a classification model
# IMPORTANT: first argument is true values, second argument is predicted values'''
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print confusion

#Metrics computed from a confusion matrix
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
#Classification Accuracy: Overall, how often is the classifier correct?
print metrics.accuracy_score(y_test, y_pred_class) #(TP + TN) / float(TP + TN + FP +FN)
#Classifcation Error(also know as  Misclassification Rate): Overall, how is the classifier incorrect?
print 1 - metrics.accuracy_score(y_test, y_pred_class)#(FP + FN) / float(TP + TN + FP +FN)
#Sensitivity (as also known as True Positive Rate or Recall): When the actual value is positive, how often is the prediction correct?
print metrics.recall_score(y_test, y_pred_class) #TP / float(TP + FN)
#Specificity: When the actual value is negative, how often is the prediction correct?
print TN / float(TN + FP)
#False Positive Rate: When te actual value is negative, how often is the prediction incorrect? 1- specificity
print FP / float(TN + FP)
#Precision: When a positive value is predicted, how often is the prediction correct?
print metrics.precision_score(y_test, y_pred_class) #TP / float(TP + FP)


'''Adjusting the classification threshold
# print the first 10 predicted responses'''
print logreg.predict(X_test)[0:10]
# print the first 10 predicted probabilities of class membership
print logreg.predict_proba(X_test)[0:10, :]
# print the first 10 predicted probabilities for class 1
print logreg.predict_proba(X_test)[0:10, 1]
# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
# plots
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.hist(y_pred_prob, bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
#plt.show()

'''Decrease the threshold for predicting diabetes in order to increase the sensitivity of the classifier
# predict diabetes if the predicted probability is greater than 0.3'''
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob, 0.3)[0]
# print the first 10 predicted probabilities
print y_pred_prob[0:10]
# print the first 10 predicted classes with the lower threshold
print y_pred_class[0:10]
# previous confusion matrix (default threshold of 0.5)
print confusion
# new confusion matrix (threshold of 0.3)
print metrics.confusion_matrix(y_test, y_pred_class)
# sensitivity has increased (used to be 0.24)
print 46 / float(46 + 16)
# specificity has decreased (used to be 0.91)
print 80 / float(80 + 50)

'''
ROC Curves and Area Under the Curve (AUC)
Question: Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?
Answer: Plot the ROC curve!
'''
# IMPORTANT: first argument is true values, second argument is predicted probabilities
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
#plt.show()

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print 'Sensitivity:', tpr[thresholds > threshold][-1]
    print 'Specificity:', 1 - fpr[thresholds > threshold][-1]

evaluate_threshold(0.5)
evaluate_threshold(0.3)

'''
AUC is the percentage of the ROC plot that is underneath the curve:
'''
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print metrics.roc_auc_score(y_test, y_pred_prob)
# calculate cross-validated AUC
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()










