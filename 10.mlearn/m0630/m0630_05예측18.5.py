from turtle import distance
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor   # 회귀
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 18.5cm 몇g일까요?
# 1개의 데이터를 산점도 그래프로 출력하시오.

# 1.데이터 불러오기
perch_length = [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
perch_weight = [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0]

# numpy타입으로 변형
length = np.array(perch_length)
weight = np.array(perch_weight)



# 2. 데이터전처리
train_data,test_data,train_label,test_label = \
    train_test_split(length,weight, random_state=42) # stratify 회귀에서는 의미가 없음.

# train_data.shape (42,) 1차원배열 -> 2차원배열
# train_data,test_data 2차원배열
  
# print(train_data) # 1차원배열 출력    
train_data = train_data.reshape(-1,1) # 2차원배열로 변경 , -1 => 모든데이터의미  
test_data = test_data.reshape(-1,1)   # 2차원배열로 변경
# print(train_data) # 2차원배열 출력

# 3. 알고리즘 선택
knr = KNeighborsRegressor()
# clf = KNeighborsClassifier()

# 4. 실습훈련
knr.fit(train_data,train_label)

# 5. 예측
result = knr.predict(test_data)
result2 = knr.predict([[18.5]])
# print("test데이터 : ",test_data)
print("예측결과 : ",result)
print("18.5 예측결과 : ",result2)

# 6. 정확도(알고리즘 성능비교)
score1 = knr.score(train_data,train_label)
score2 = knr.score(test_data,test_label)
print("train 예측 : ",score1)
print("test 예측 : ",score2)

# 7. 오차범위
mae = mean_absolute_error(test_label,result)
print("오차범위 : ", mae)

# knn이웃하는 데이터 - train_data,train_label
distances,indexs = knr.kneighbors([[18.5]])

# 그래프 
plt.scatter(train_data,train_label)
plt.scatter(train_data[indexs],train_label[indexs],marker="D")
plt.scatter(18.5,79.6,marker="^")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()