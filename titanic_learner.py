import numpy as np
import pandas as pd
# import sklearn
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('cleaned_train_data.csv')
test_df = pd.read_csv('cleaned_test_data1.csv')

test_df = test_df.fillna(0)

train_data = train_df.values
test_data = test_df.values
# train_column_names = list(train_df.columns.values)
# test_column_names = list(test_df.columns.values)

# Train columns: ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'EmbarkedInt', 'AgeFill', 'AgeIsNull', 'FamilySize']
# Test columns: ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'EmbarkedInt', 'AgeFill', 'AgeIsNull', 'FamilySize']

#Zero Indexed 

# Training data
train_columns = 2,6,8 #Class, Age, Gender
X_train = train_data[:,train_columns]
y_train = train_data[:,1]

# Testing data
test_columns = 1,5,7
X_test = test_data[:,test_columns]
# y_test = 

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

pred_df = pd.DataFrame({"PassengerID":pd.Series(test_data[:,0], dtype=int), 
						"Survived":pd.Series(pred, dtype=int)})


pred_df.to_csv('RandomForest_predictions.csv', index=False)