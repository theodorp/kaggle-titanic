import pandas as pd
import numpy as np

def data_cleaner(filepath):

    df = pd.read_csv(filepath)

    # Gender: Females = 0, Males = 1
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1})

    # EmbarkedInt: C = 0, Q = 1, S = 2, nan = 0
    df['EmbarkedInt'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3}).fillna('0')

    # AgeIsNull: 1 = Age was NaN, 0 = Age was present
    df['AgeIsNull'] = df['Age'].isnull().astype(int)

    # AgeFill: 
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & \
                                  (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                    'AgeFill'] = median_ages[i,j]
    
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # FareFill
    df['FareFill'] = df['Fare'].fillna('0')
    
    median_fare = df[['Pclass','Fare']][df.Fare > 0].groupby('Pclass').median()
    
    for i in range(1,4):
        df.loc[(df.Fare == 0) & (df.Pclass == i), 'FareFill'] = median_fare.Fare[i]    
    
    pd.options.mode.chained_assignment = None

    # Title:
    df['Title'] = df['Name']
    df['Title'] = df['Title'].map(lambda x: x.rsplit(',')[1].rsplit('.')[0].strip())
    df['TitleInt'] = df['Title'].apply(lambda x: 0 if x in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Mr'] else 1)

    # df['classXfare'] = df['Pclass'].astype(float)*df['FareFill'].astype(float)
    # df['logClass'] = df['Pclass'].apply(lambda x: (2-np.log(x)))

    df = df.drop(['Age', 'Name', 'Fare', 'Sex', 'Ticket', 'Cabin', 'AgeIsNull',\
                  'SibSp', 'Parch', 'EmbarkedInt', 'Embarked', 'Title', 'TitleInt'], 1)
    
    return df