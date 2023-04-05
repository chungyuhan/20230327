import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/train_titanic.csv")
df.head()
df.info()

df[['Passenger_Group','Passenger_Number']] = df.PassengerId.str.split('_',expand = True).astype(int)
df.set_index('PassengerId', inplace=True)

df.isnull().sum()
df.describe()

numerical_columns = ['Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_columns = ['VIP','CryoSleep']
encoded_columns = ['HomePlanet','Destination']

from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='mean')

for i in numerical_columns:
    df[i] = impute.fit_transform(df[[i]])

impute = SimpleImputer(strategy='most_frequent')
for i in categorical_columns:
    df[i] = impute.fit_transform(df[[i]])

df['Cabin'].fillna('T/0/P',inplace=True)

df.info()
df.drop(['Name'], axis=1, inplace=True)

df['CryoSleep'] = df['CryoSleep'].astype(bool)

from sklearn.preprocessing import normalize

for i in numerical_columns:
    normalize(df[[i]])
df.describe()

sns.countplot(data=df, x='HomePlanet', hue='Transported')
sns.countplot(data=df, x='Destination', hue='Transported')

df['Age'] = df['Age'].astype(int)
sns.histplot(df, x='Age', hue='Transported')

df_train = pd.DataFrame(df[['VIP','Transported']].astype(bool),columns=['VIP','Transported'])
sns.countplot(data=df_train, x='VIP', hue='Transported')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for i in encoded_columns:
    df[i] = encoder.fit_transform(df[i])

df.info()

X = df.drop(['Transported','Cabin'],axis=1)
y = df['Transported']
X.info()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(max_iter=200)

# lr.fit(X_train, y_train)

# predictions = lr.predict(X_test)

# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# accuracy_score(y_test, predictions)
# recall_score(y_test, predictions)
# precision_score(y_test, predictions)

# pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predict not Transported', 'Predict Transported'],index=['True not Transported','True Transported'])

# import joblib
# joblib.dump(lr,'Space-LR-20230404.pkl',compress=3)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(X,y)
clf.score(X,y)

df_test = pd.read_csv("test-2.csv")

impute = SimpleImputer(strategy='mean')
for i in numerical_columns:
    df_test[i] = impute.fit_transform(df_test[[i]])

df_test.isnull().sum()
df_test[['Passenger_Group','Passenger_Number']]=df_test['PassengerId'].str.split('_',expand=True)
df_test.set_index('PassengerId',inplace=True)

df_test

impute = SimpleImputer(strategy='most_frequent')
for i in categorical_columns:
    df_test[i] = impute.fit_transform(df_test[[i]])

df_test.isnull().sum()

for i in encoded_columns[:-1]:
    df_test[i] = impute.fit_transform(df_test[[i]])

df_test.isnull().sum()

for i in encoded_columns[:-1]:
    df_test[i] = encoder.fit_transform(df_test[i])

df_test.drop(['Cabin','Name'],axis=1,inplace=True)

df_test['Age'].describe()

X.columns
df_test.columns

for i in encoded_columns:
    df_test[i] = encoder.fit_transform(df_test[[i]])

pred = clf.predict(df_test)
submission = pd.DataFrame(df_test.index,columns=['PassengerId'])
submission['Transported'] = pred
submission.to_csv('submission.csv',index=False)