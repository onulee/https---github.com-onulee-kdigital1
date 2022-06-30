from sklearn.neighbors import KNeighborsRegressor   # knn회귀
from sklearn.linear_model import LinearRegression   # 선형회귀
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 50cm,100cm 농어 각각 몇g일까요?  
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
    

# print(train_data) # 1차원배열 출력    
train_data = train_data.reshape(-1,1) # 2차원배열로 변경 , -1 => 모든데이터의미  
test_data = test_data.reshape(-1,1)   # 2차원배열로 변경
# print(train_data) # 2차원배열 출력

# 다항회귀 데이터처리
train_poly = np.column_stack((train_data**2,train_data))
test_poly = np.column_stack((test_data**2,test_data))

# 3. 알고리즘 선택
lr = LinearRegression()
# knr = KNeighborsRegressor()

# 4. 실습훈련- 다항회귀 데이터
lr.fit(train_poly,train_label)

# LinearRegression -> 기울기, y절편
# print("기울기 : ",lr.coef_) # 기울기 lr.coef_[0],lr.coef_[1]
# print("y절편 : ",lr.intercept_) #  y절편

# 5. 예측 - 데이터(제곱데이터,데이터)
result = lr.predict([[50**2,50]])    # 1573
result2 = lr.predict([[100**2,100]]) #8103
# print("test데이터 : ",test_data)
print("50cm 예측결과 : ",result)
print("100cm 예측결과 : ",result2)

# 6. 정확도(알고리즘 성능비교)
score1 = lr.score(train_poly,train_label)
score2 = lr.score(test_poly,test_label)
print("train 예측 : ",score1)
print("test 예측 : ",score2)

# 구간별 직선을 그리기 위해 범위를 정수 배열로 생성
point = np.arange(15,100)

# 그래프 
plt.scatter(train_data,train_label)
# # 기울기와 y절편을 이용한 선그래프
# # y = ax(기울기제곱) + b(기울기) + c(y절편)
# 기울기 lr.coef_[0],lr.coef_[1]
# x축 15~100 y축
plt.plot(point,lr.coef_[0]*point**2+lr.coef_[1]*point+lr.intercept_)
plt.scatter(50,1573,marker="D")
plt.scatter(100,8103,marker="^")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()