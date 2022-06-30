from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 길이, 높이, 폭 데이터 가져오기
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy() #numpy타입으로 변경

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

# 훈련데이터, 테스트데이터 분류
train_data, test_data, train_label, test_label = train_test_split(
    perch_full, perch_weight, random_state=42)

# 다항 모델 적용
# # 기본값 degree=2, 제곱
# include_bias 기본적으로 1적용 - True
poly = PolynomialFeatures(degree=2,include_bias=False)

# poly.fit([[2,3]])
# # 1(bias),2,3,2**2,2*3,3**2 -> 형태로 구성
# # [[1. 2. 3. 4. 6. 9.]]
# train_poly = poly.transform([[2,3]])

poly.fit(train_data)
train_poly = poly.transform(train_data)
# 특성이 3개(length,height,width)에서 9개 늘어남, 행은 42개 (42,9)
# train_poly형태 출력
print(train_poly.shape)

# 특성(변수)이 구성된 형태를 알려줌.
print(poly.get_feature_names())
# ['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']

test_poly = poly.transform(test_data)

# LinearRegression 모델 적용
lr = LinearRegression()
lr.fit(train_poly,train_label)
print('정확도 : ',lr.score(train_poly,train_label))
print('정확도 : ',lr.score(test_poly,test_label))