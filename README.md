# 20230327
1.我參加的是預測在太空船鐵達尼號與時空異常碰撞期間，乘客是否被傳送到了另一個維度的競賽，選擇這一個競賽的原因是他與上課教的一樣都是資料分析及預測，資料內容也與上課教的相似，對於這個文組生來說，雖然很想挑戰別的競賽，例如老師上課提到的房價預測，但實際在研究如何寫程式碼的時候，有點力不從心，因此最終還是選擇較為簡單的太空船鐵達尼號，但我還是有參考他人的作品來撰寫。這項競賽有龐大的資料及，其中包含「PassengerId、HomePlanet、CryoSlee、Cabin、Destination、Age、VIP、RoomService、FoodCourt、ShoppingMall、Spa、VRDeck、Name、Transported」每筆接近8700份資料，並且多筆資料都有缺失，而test.cvs裡只有三分之一的乘客紀錄，因此結果需顯示「PassengerId」和「Transported」，其中「Transported」須以True和False的方式呈現。
2.我是根據Kaggle上的其他參賽者來撰寫，隨然原先也想趙老師上課的方式，但不知如何下手，因此是照著他人的程式碼，在遇到不懂的程式碼時，我會向ChatGPT詢問，因此這次的期末作業我有大量理解是從ChatGPT的解釋而來，我認為這對我去學習、看懂程式碼的幫助很多，以下是我實作的成果及程式碼理解：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
讀取訓練檔案，並初步觀察資料
df = pd.read_csv("data/train_titanic.csv")
df.head()
df.info()
PassengerId以‘＿’當區分的分割點，把資料分割成兩的部分，並且分別放入新創的欄位「'Passenger_Group','Passenger_Number'」的裡面，而‘astype(int)’是為了將‘PassengerId’的值轉換為整數數據類型，以便於之後的分析，最後將‘PassengerId‘設置為索引
df[['Passenger_Group','Passenger_Number']] = df.PassengerId.str.split('_',expand = True).astype(int)
df.set_index('PassengerId', inplace=True)
觀察每筆資料缺失的數量
df.isnull().sum()
df.describe()
將每個欄位按照數值、類別和對象進行分類，這樣做是為了後續可以針對每個類別的數據進行不同的處理。
numerical_columns = ['Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_columns = ['VIP','CryoSleep']
encoded_columns = ['HomePlanet','Destination']
使用’SimpleImputer‘來填補缺失值，而在這個部分是幫’numerical_columns‘以平均值來填補缺失值
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='mean')
for i in numerical_columns:
    df[i] = impute.fit_transform(df[[i]])
與上一個步驟一樣，只是’categorical_columns‘的缺失值改為使用該列中出現最多的值來填補
impute = SimpleImputer(strategy='most_frequent')
for i in categorical_columns:
    df[i] = impute.fit_transform(df[[i]])
將’Cabin‘的缺失值以'T/0/P'來填補（此為上課教的填補方法）
df['Cabin'].fillna('T/0/P',inplace=True)
df.info()
丟掉不會用到的'Name'欄位
df.drop(['Name'], axis=1, inplace=True)
將'CryoSleep'的資料轉換為布林類型(bool)，即 True 或 False的形式
df['CryoSleep'] = df['CryoSleep'].astype(bool)







