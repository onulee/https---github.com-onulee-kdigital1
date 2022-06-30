from sklearn.linear_model import LinearRegression   # 선형회귀
from sklearn.preprocessing import PolynomialFeatures # 다항특성만들기
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 길이, 높이, 폭 데이터 가져오기
# 1. 데이터 가져오기
df = pd.read_csv('10.mlearn/m0630/perch_full.csv')
perch_full = df.to_numpy() #numpy타입으로 변경 (56,3)
# 무게 데이터
perch_weight = np.array(
[5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
1000.0, 1000.0]
)

# 2. 데이터 전처리 (56,3)
train_data,test_data,train_label,test_label = \
    train_test_split(perch_full,perch_weight)

# 다중회귀일때 다항모델 적용
# 알고리즘 선택
# 기본값 degree=2
poly = PolynomialFeatures(degree=2,include_bias=False)
poly.fit(train_data)
train_poly = poly.transform(train_data)
test_poly = poly.transform(test_data)
new_poly = poly.transform([[18.6,5.55,3.04]])

# print(train_poly.shape) # (56,3) -> (42,9)
# print(test_poly.shape)  # (14,9)
# 특성이 구성된 형태를 알려줌
# print(poly.get_feature_names())

# 정규화작업
# ss = StandardScaler()
# ss.fit(train_poly)
# train_scaled = ss.transform(train_poly)
# test_scaled = ss.transform(test_poly)

# 2차원방정식 변환
# train_poly = np.column_stack(train_data**2,train_data)
# test_poly = np.column_stack(test_data**2,test_data)

# 3. 알고리즘 선택
lr = LinearRegression()

# 4. 실습훈련
lr.fit(train_poly,train_label)

# 5. 예측
result = lr.predict(new_poly)
print("무게예측 : ",result)

# 6. 정확도 확인
score1 = lr.score(train_poly,train_label)
score2 = lr.score(test_poly,test_label)
print("train 정확도 : ",score1)
print("test 정확도 : ",score2)
