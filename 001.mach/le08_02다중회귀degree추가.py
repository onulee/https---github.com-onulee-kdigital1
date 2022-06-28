from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

#----------------------------------------------------
# [ degree=5 ] 변경
# 다중모델 degree추가 5로 변경 -> test_poly,test_label - 음수로 떨어짐.
poly = PolynomialFeatures(degree=5,include_bias=False)
poly.fit(train_data)
train_poly = poly.transform(train_data)
test_poly = poly.transform(test_data)
# 규제,정규화작업 진행
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀 사용 - 규제,정규화 작업후 릿지회귀 사용함.
ridge = Ridge()
ridge.fit(train_scaled,train_label)

print('정확도 : ',ridge.score(train_scaled,train_label))
print('정확도 : ',ridge.score(test_scaled,test_label))