#!/usr/bin/env python

#Scikit-learn Tutorial: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

from matplotlib import pyplot as plt
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import statsmodels

# Load dataset - using pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Dataset Summary:

# Shape
print(dataset.shape)

# "Head" Data View
print(dataset.head(20))

# Statistics
print(dataset.describe())

# Class Distribution
print(dataset.groupby('class').size())

# Dataset Visualization:

# Box + Whisker Plots
#dataset.plot(kind = "box", subplots = True, layout = (2,2), sharex = False, sharey = False)
#plt.show()

# Histograms
#dataset.hist()
#plt.show()

# Multivariate Plots:
# Scatter Plot Matrix
#pandas.plotting.scatter_matrix(dataset)
#plt.show()

# Evaluate ML Algorithms:

# Create Validation Set
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
# Stratified (equal class distribution) 10-fold CV for model accuracy evaluation
# Random state = random seed in R
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size = 0.20, random_state = 1)

# Build Models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# Evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms - SVM wins for this dataset
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# Make Predictions:

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

