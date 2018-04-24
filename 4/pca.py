# -*- coding: utf-8 -*-
#Ashwini Kulkarni
#import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


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

#from sklearn.preprocessing import StandardScaler
#X_std = StandardScaler().fit_transform(X)


print("************** With PCA Component 3 ******************")

# 3 components
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
X_pca = sklearn_pca.fit_transform(X)

model.fit(X_pca, y)






# make predictions
expected = y
predicted = model.predict(X_pca)
print( "%d datapoints out of %d are not correctly predicting using '3 components PCA'\n"%((expected != predicted).sum(), X_pca.shape[0]))
print("\n Classification Report :-")
print(metrics.classification_report(expected, predicted))




# 2 components

print("************** With PCA Component 2 ******************")


sklearn_pca = sklearnPCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X)
#print(Y_sklearn1)
model.fit(X_pca, y)

# make predictions
expected = y
predicted = model.predict(X_pca)
print( "%d datapoints out of %d are not correctly predicting using '2 components PCA'.\n"%((expected != predicted).sum(),X_pca.shape[0]))
print("\n Classification Report :-")
print(metrics.classification_report(expected, predicted))



print("************** With PCA Component 1 ******************")
# 1 components

sklearn_pca = sklearnPCA(n_components=1)
X_pca = sklearn_pca.fit_transform(X)

model.fit(X_pca, y)

# make predictions
expected = y
predicted = model.predict(X_pca)
print( "%d datapoints out of %d are not correctly predicting using '1 components PCA'.\n"%((expected != predicted).sum(),X_pca.shape[0]))
print("\n Classification Report")
print(metrics.classification_report(expected, predicted))



# -*- coding: utf-8 -*-

