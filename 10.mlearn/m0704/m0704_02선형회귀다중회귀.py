from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# 선형회귀 - 다중회귀
#  length,height,width -> length**2,length,height**2,height,width**2,width
df = pd.read_csv('10.mlearn/m0630/perch_full.csv')

perch_weight = np.array(
[5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
1000.0, 1000.0]
)
# df['weight'] = perch_weight
# df.to_csv('10.mlearn/m0630/perch_full2.csv',encoding='utf-8-sig',index=False)
# df = pd.read_csv('10.mlearn/m0630/perch_full2.csv')
# print(df)

# numpy배열타입으로 변경, pd.to_numpy(), np.array()
perch_full = df.to_numpy()
# perch_full = np.array(df)

# 1. 데이터 전처리
train_data,test_data,train_label,test_label = \
    train_test_split(perch_full,perch_weight)

# 다중회귀 - 다항회귀
# train_data = np.column_stack((train_data***2,train_data**2,train_data)) 
# degree : 다차원방정식 생성 - degree=2
poly = PolynomialFeatures(degree=5,include_bias=False)
poly.fit(train_data)
train_poly = poly.transform(train_data) # 2차원방정식으로 변형
# poly.fit_transform(train_data)
test_poly = poly.transform(test_data)
# print(poly.get_feature_names())  #특성의 구성
# print(train_data.shape)
# print(train_poly.shape)
new_poly = poly.transform([[30.4,8.89,4.22]])
   
# 2. 알고리즘 선택
# 규제 - 릿지회귀
lr = LinearRegression()

# 3. 실습훈련
lr.fit(train_poly,train_label)
print(lr.coef_,lr.intercept_)

# 4. 예측 30.4,8.89,4.22 특성3개
result = lr.predict(new_poly)    
predict = lr.predict(test_poly)

# 5. 정확도체크
score1 = lr.score(train_poly,train_label)
score2 = lr.score(test_poly,test_label)
print("예측1 : ",score1)
print("예측2 : ",score2)  # 마이너스 값이 나타남. z= a x**2

# 6. 오차범위
mae = mean_absolute_error(test_label,predict)
print("오차범위 : ",mae)