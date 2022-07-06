from sklearn.linear_model import LogisticRegression, SGDClassifier # 확률적경사하강법
from sklearn.model_selection import train_test_split # train,test
from sklearn.preprocessing import PolynomialFeatures, StandardScaler     # 정규화,표준화작업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터불러오기 0:red wine, 1:white wine
# 특성 : alcohol,sugar,pH  클래스 : 2개 class
# wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine = pd.read_csv('10.mlearn/m0706/wine.csv')
print(wine.info())
data = wine[['alcohol','sugar','pH']].to_numpy()
label = wine['class'].to_numpy()

# 로지스틱 회귀 - 정확도를 출력하시오.
# 데이터 전처리
train_data, test_data, train_label, test_label = train_test_split(
    data, label, random_state=42)
# x_data, x_data, y_label, y_label = train_test_split(
#     x, y, random_state=42)

# 정규화,표준화작업
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)

# clists = [0.001,0.01,0.1,1,10,100]
# train_score=[]
# test_score=[]
# for clist in clists:
#    lr = LogisticRegression(C=clist)
#    lr.fit(train_scaled,train_label)
#    train_score.append(lr.score(train_scaled,train_label))  
#    test_score.append(lr.score(test_scaled,test_label))  
   
# plt.plot(np.log10(clists),train_score)   
# plt.plot(np.log10(clists),test_score)   
# plt.show()

# 알고리즘 선택
lr = LogisticRegression()
lr.fit(train_scaled,train_label)
print("train score : ",lr.score(train_scaled,train_label))
print("test score : ",lr.score(test_scaled,test_label))

