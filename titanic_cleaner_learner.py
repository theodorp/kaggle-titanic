import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

from titanic_cleaner import data_cleaner

train_df = data_cleaner('data/train.csv')
test_df = data_cleaner('data/test.csv')

# print(train_df.columns.values)
# print(train_df.tail(3))
# print(test_df.head())

features = ['Pclass', 'Gender', 'AgeFill']

X = train_df.ix[:,features].values
y = train_df.ix[:, 'Survived'].values

from sklearn.cross_validation import cross_val_score

# Let's do a 2-fold cross-validation of the SVC estimator
print(cross_val_score(SVC(), X, y, cv=5, scoring='accuracy'))



# X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.33, random_state=42)

# skf = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=42)

# for train_index, test_index in skf:
# 	X_train, X_cv = X[train_index], X[test_index]
# 	y_train, y_cv = y[train_index], y[test_index]



# 	clf = RandomForestClassifier(n_estimators=30)

# 	clf.fit(X_train, y_train)

# 	pred = clf.predict(X_cv)

# 	print('clf.score: ', clf.score(X_cv, y_cv))
# 	print('accuracy_score: ', accuracy_score(y_cv, pred))