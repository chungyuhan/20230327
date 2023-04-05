import joblib
model_pretrained = joblib.load('Space-LR-20230404.pkl')
import pandas as pd

df_test = pd.read_csv("test-2.csv")

df_test.isnull().sum()
df_test.head

df_test[['Passenger_Group','Passenger_Number']]=df_test['PassengerId'].str.split('_',expand=True)
df_test.set_index('PassengerId',inplace=True)

numerical_columns = ['Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_columns = ['VIP','CryoSleep']
encoded_columns = ['HomePlanet','Destination']

from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='mean')
for i in numerical_columns:
    df_test[i] = impute.fit_transform(df_test[[i]])

impute = SimpleImputer(strategy='most_frequent')
for i in categorical_columns:
    df_test[i] = impute.fit_transform(df_test[[i]])

df_test.drop(['Name'], axis=1, inplace=True)

df_test['CryoSleep'] = df_test['CryoSleep'].astype(bool)

from sklearn.preprocessing import normalize
for i in numerical_columns:
    normalize(df_test[[i]])

df_test['Age'] = df_test['Age'].astype(int)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in encoded_columns:
    df_test[i] = encoder.fit_transform(df_test[i])


df_test.isnull().sum()

df_test.drop(['Cabin'], axis=1, inplace=True)


predictions2 = model_pretrained.predict(df_test)
predictions2

forSubmissionDF = pd.DataFrame(columns=['PassengerId','Transported'])
forSubmissionDF
forSubmissionDF['PassengerId'] = range(8693,4277)
forSubmissionDF['Transported'] = predictions2
forSubmissionDF


