# -*- coding: utf-8 -*-
"""
@author: Ashwini Kulkarni
"""

# -*- coding: utf-8 -*-

#import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from sklearn.datasets import load_iris
data = load_iris()

print("Flowers:-" ,data.target_names)

# split data table into data X and class labels y

X = data.data
y = data.target


# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(X, y)

print("Total No of Data Points %d" %X.shape[0])
print("**************** With All Features *******************")
# make predictions
expected = y
predicted = model.predict(X)
print( "%d datapoints out of %d are not correctly predicting with all feature consideration \n"%((expected != predicted).sum(),X.shape[0]))
print("\n Classification Report :-")
print(metrics.classification_report(expected, predicted))



print("************** With LDA Component 3 ******************")

# 3 components

sklearn_lda = lda(n_components=3)
X_lda = sklearn_lda.fit(X, y).transform(X)


model.fit(X_lda, y)

# make predictions
expected = y
predicted = model.predict(X_lda)
print( "%d datapoints out of %d are not correctly predicting using '3 components LDA'\n"%((expected != predicted).sum(), X_lda.shape[0]))
print("\n Classification Report :-")
print(metrics.classification_report(expected, predicted))




# 2 components

print("************** With LDA Component 2 ******************")


sklearn_lda = lda(n_components=2)
X_lda = sklearn_lda.fit(X, y).transform(X)
#print(Y_sklearn1)
model.fit(X_lda, y)

# make predictions
expected = y
predicted = model.predict(X_lda)
print( "%d datapoints out of %d are not correctly predicting using '2 components LDA'.\n"%((expected != predicted).sum(),X_lda.shape[0]))
print("\n Classification Report :-")
print(metrics.classification_report(expected, predicted))



print("************** With LDA Component 1 ******************")
# 1 components

sklearn_lda = lda(n_components=1)
X_lda = sklearn_lda.fit(X, y).transform(X)

model.fit(X_lda, y)

# make predictions
expected = y
predicted = model.predict(X_lda)
print( "%d datapoints out of %d are not correctly predicting using '1 components LDA'.\n"%((expected != predicted).sum(),X_lda.shape[0]))
print("\n Classification Report")
print(metrics.classification_report(expected, predicted))



# -*- coding: utf-8 -*-

