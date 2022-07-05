from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit                      # 시그모이드함수
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 도미데이터 = 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 
33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 
610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 = 14개 1
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 전체 49개, 도미 35개, 빙어 14개 len()
length = bream_length + smelt_length  
weight = bream_weight + smelt_weight

data = np.column_stack((length,weight))
label = np.concatenate((np.array(['도미']*35),np.array(['빙어']*14)))

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 정규화표준화작업
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)
new_scaled1 = ss.fit_transform([[30,600]])
new_scaled2 = ss.fit_transform([[25,150]])

# 알고리즘 선택
lr = LogisticRegression()

# 실습훈련
lr.fit(train_scaled,train_label)

# 예측
result = lr.predict(new_scaled1)
print("30,600의 분류 : ",result)
result2 = lr.predict(new_scaled2)
print("25,150의 분류 : ",result2)
print("-"*50)
print(lr.predict(test_scaled[:5]))
proba = lr.predict_proba(test_scaled[:5])
print(proba)

# 정확도
score1 = lr.score(train_scaled,train_label)
score2 = lr.score(test_scaled,test_label)
print("정확도1 : ",score1)
print("정확도2 : ",score2)

# z값 출력
decisions = lr.decision_function(test_scaled[:5])
print(decisions)

# 시그모이드함수
print(expit(decisions))
