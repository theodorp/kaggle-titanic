# !/usr/bin/python

# The first thing to do is to import the relevant packages
# that I will need for my script, 
# these include the Numpy (for maths and arrays)
# and csv for reading and writing csv files
# If i want to use something from this I need to call 
# csv.[function] or np.[function] first

import pandas as pd
import numpy as np
import pylab as P

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('data/test.csv', header=0)

# Add Gender column female = 0 and male = 1
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# S = 0, C = 1, Q = 2
df['EmbarkedInt'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

median_ages = np.zeros((2,3))

for i in range(0, 2):
	for j in range(0, 3):
		median_ages[i,j] = df[(df['Gender'] == i) & \
			(df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0,2):
	for j in range(0, 3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
# df = df.dropna()

# train_data = df.values
# df_column_names = list(df.columns.values)

df.to_csv('cleaned_test_data1.csv', index=False)