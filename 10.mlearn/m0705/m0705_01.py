from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit                      # 시그모이드함수
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 선형회귀기반 - 기울기,y절편 z = ax + b , 예측값 = 기울기*특성1+y절편
# 로지스틱회귀 - class 2개, class 여러개 분류 : 정확도 %

# 데이터 불러오기 [159 rows x 6 columns]
df_fish = pd.read_csv('10.mlearn/m0705/fish.csv')
data = df_fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
label = df_fish['Species'].to_numpy()  #7가지['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt'] 
# print(df_fish['Species'].unique())
# ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt'] class 7개
# ['Weight','Length','Diagonal','Height','Width'] 특성 5개

# # 데이터전처리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# Bream,Smelt 도미,빙어
index1 = (train_label == 'Bream') | (train_label == 'Smelt')
index2 = (test_label == 'Bream') | (test_label == 'Smelt')
train_bream_smelt = train_data[index1] # 119 -> 33개 (Bream,Smelt)
train_label2 = train_label[index1]     #        33개
# print(train_bream_smelt.shape)  
test_bream_smelt = test_data[index2] # 40 -> 16개 (Bream,Smelt)
test_label2 = test_label[index2]     #       16개  
# print(test_label2)
# print(test_label2.shape)

# 정규화,표준화작업
ss = StandardScaler() 
train_scaled = ss.fit_transform(train_bream_smelt)
test_scaled = ss.fit_transform(test_bream_smelt)


# 알고리즘 선택
lr  = LogisticRegression()
# 실습훈련 - train
lr.fit(train_scaled,train_label2)
print("-"*50)
print(lr.coef_,lr.intercept_)
# 기울기 5개 : 특성 5개 z = ax1 + bx2 + cx3 + dx4 + ex5 + y절편  
print("-"*50)

# 예측
result = lr.predict(test_scaled[:5])
print("결과값 : ",result)
result2 = lr.predict_proba(test_scaled[:5])
print(result2)

# z값을 출력 z = ax1 + bx2 + cx3 + dx4 + ex5 + y절편
decisions = lr.decision_function(test_scaled[:5])
print(decisions)

# 시그모이드함수 적용
print(expit(decisions))


# 정확도
# score1 = lr.score()
# score2 = lr.score()
# print("score1정확도 : ",score1)
# print("score2정확도 : ",score2)
