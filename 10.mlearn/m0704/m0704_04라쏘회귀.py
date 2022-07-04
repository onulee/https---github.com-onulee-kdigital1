from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
    train_test_split(perch_full,perch_weight,random_state=42)

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
  

# 정규화,표준화 작업
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
new_scaled = ss.transform(new_poly)
# ss.fit_transform(test_poly)
   
# 2. 알고리즘 선택 : 규제 - 릿지회귀 alpha값 확인
# 규제 - 릿지회귀
lr = LinearRegression()
# Ridge규제 : alpha-규제강도, 값이 높을수록 강도가 강함, 값이 낮을수록 강도가 약함.

# train_score=[]
# test_score=[]
# alpha_list = [0.01,0.1,1,10,100]
# for list in alpha_list:
#     ridge = Lasso(alpha=list)
#     # 3. 실습훈련
#     ridge.fit(train_scaled,train_label)
#     print(ridge.coef_,ridge.intercept_)
#     # 5. 정확도체크
#     train_score.append(ridge.score(train_scaled,train_label))
#     test_score.append(ridge.score(test_scaled,test_label))

# # 선그래프
# plt.plot(np.log10(alpha_list),train_score)
# plt.plot(np.log10(alpha_list),test_score)
# plt.xlabel('alpha')
# plt.ylabel('score')
# plt.show()


# 특성값을 제거해서 규제
lasso = Lasso(alpha=10)
# 3. 실습훈련
lasso.fit(train_scaled,train_label)
# print(ridge.coef_,ridge.intercept_)

# 4. 예측 30.4,8.89,4.22 특성3개
result = lasso.predict(new_scaled)    
predict = lasso.predict(test_scaled)

# 5. 정확도
score1 = lasso.score(train_scaled,train_label)
score2 = lasso.score(test_scaled,test_label)
print("score1예측 : ",score1)
print("score2예측 : ",score2)

print("0인 특성값 :",np.sum(lasso.coef_ == 0))

# 6. 오차범위
mae = mean_absolute_error(test_label,predict)
print("오차범위 : ",mae)