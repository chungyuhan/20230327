import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 讀取訓練檔案，並初步觀察資料
df = pd.read_csv("data/train_titanic.csv")
df.head()
df.info()
# PassengerId以‘_’當區分的分割點，把資料分割成兩的部分，並且分別放入新創的欄位「'Passenger_Group','Passenger_Number'」的裡面，
# 而‘astype(int)’是為了將‘PassengerId’的值轉換為整數數據類型，以便於之後的分析，最後將‘PassengerId‘設置為索引
df[['Passenger_Group','Passenger_Number']] = df.PassengerId.str.split('_',expand = True).astype(int)
df.set_index('PassengerId', inplace=True)
# 觀察每筆資料缺失的數量
df.isnull().sum()
df.describe()
# 將每個欄位按照數值、類別和對象進行分類，這樣做是為了後續可以針對每個類別的數據進行不同的處理。
numerical_columns = ['Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_columns = ['VIP','CryoSleep']
encoded_columns = ['HomePlanet','Destination']
# 使用’SimpleImputer‘來填補缺失值，而在這個部分是幫’numerical_columns‘以平均值來填補缺失值
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='mean')
for i in numerical_columns:
    df[i] = impute.fit_transform(df[[i]])
# 與上一個步驟一樣，只是’categorical_columns‘的缺失值改為使用該列中出現最多的值來填補
impute = SimpleImputer(strategy='most_frequent')
for i in categorical_columns:
    df[i] = impute.fit_transform(df[[i]])
# 將’Cabin‘的缺失值以'T/0/P'來填補（此為上課教的填補方法）
df['Cabin'].fillna('T/0/P',inplace=True)
df.info()
# 丟掉不會用到的'Name'欄位
df.drop(['Name'], axis=1, inplace=True)
# 將'CryoSleep'的資料轉換為布林類型(bool)，即 True 或 False的形式
df['CryoSleep'] = df['CryoSleep'].astype(bool)
# 藉由‘normalize’為‘numerical_columns’進行資料歸一化，從ChatGPT上得知歸一化的用意為：
# 避免不同特徵的量級差異對建模結果造成影響，因此通過減去平均值並除以標準差來完成
from sklearn.preprocessing import normalize
for i in numerical_columns:
    normalize(df[[i]])
df.describe()
# 使用‘countplot‘來繪製圖，此圖是繪製出’HomePlanet‘變數的分布圖，同時按照’Transported‘變數的值進行著色，
# 此外這一個部分原先使用參考版本時有失敗（原先程式碼：sns.countplot(train_data,x='HomePlanet',hue='Transported')），
# 因此有交給ChatGPT進行修改，他修改的部分是將資料集'df'傳遞給'data'參數，
# 並將‘Transported’變數傳遞給‘x’參數，將‘HomePlanet’變數傳遞給‘hue’參數，
# 其解釋為：‘df‘應該被指定給’data‘參數而不是’x‘參數，並且’HomePlanet‘是一個類別變數，不能直接傳遞給’x‘參數
sns.countplot(data=df, x='HomePlanet', hue='Transported')
# 與上述解釋相同，只是改為觀察‘Destination’變數的分佈圖
sns.countplot(data=df, x='Destination', hue='Transported')
# 使用’astype‘將'Age'轉換為整數型態
df['Age'] = df['Age'].astype(int)
# 使用‘histplot’來繪製直方圖，'Age'為繪製的變量，'Transported'進行分組，其中每個’Transported‘值的分布用不同的顏色表示
sns.histplot(df, x='Age', hue='Transported')
# 建立了一個新的 Pandas DataFrame‘df_train'，其中包含'VIP'、'Transported'兩個欄位，並使用‘astype’來將它們的資料類型轉換為布林值（Bool）
df_train = pd.DataFrame(df[['VIP','Transported']].astype(bool),columns=['VIP','Transported'])
# 一樣使用‘countplot’來繪圖
sns.countplot(data=df_train, x='VIP', hue='Transported')
# 此處是使用‘LabelEncoder’來將’encoded_columns‘（標稱型特徵）進行編碼處理，而標稱型特徵是指那些沒有任何數值含義的特徵，例如顏色、地點等
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in encoded_columns:
    df[i] = encoder.fit_transform(df[i])
df.info()
# 設定‘Ｘ’並將'Transported'、'Cabin'丟掉，而’y‘代表'Transported'
X = df.drop(['Transported','Cabin'],axis=1)
y = df['Transported']
X.info()

# 此處的程式碼是我嘗試將上課所學的方式來預測，雖然在一開始沒有問題，但在訓練正式test.cvs時有問題，因此這部分我先註解在這裡
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

# 因上述的程式碼在後續操作有問題，因此我最後還是以參考資料來完成整個競賽
# 此部分是使用‘sklearn’庫中的‘RandomForestClassifier’，構建了一個基於隨機森林的分類器模型，而ChatGPT解釋隨機森林是一種集成學習方法，
# 通過隨機選擇數據樣本和特徵來構建多個決策樹，最終將它們組合成一個分類器，而‘score()’方法會將X輸入到模型中進行預測，然後將預測結果與y進行比較，這裡給的分數為0.9853905441159554
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(X,y)
clf.score(X,y)
# 讀取test檔案
df_test = pd.read_csv("test-2.csv")
# 觀察test檔案，將test的資料進行修改，方式與train一樣
df_test.info()
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
# 檢查‘Ｘ’與‘df_test’是否一致
X.columns
df_test.columns
for i in encoded_columns:
    df_test[i] = encoder.fit_transform(df_test[[i]])
# 最後使用之前訓練好的隨機森林分類器來對測試集進行預測，然後將預測結果轉換成指定格式的DataFrame，並將其寫入一個csv文件中，其命名為‘submission’。
pred = clf.predict(df_test)
submission = pd.DataFrame(df_test.index,columns=['PassengerId'])
submission['Transported'] = pred
submission.to_csv('submission.csv',index=False)