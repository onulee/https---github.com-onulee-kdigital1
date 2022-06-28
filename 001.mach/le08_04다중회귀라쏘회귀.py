from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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
# [ 규제 강도 찾기 ]

poly = PolynomialFeatures(degree=5,include_bias=False)
poly.fit(train_data)
train_poly = poly.transform(train_data)
test_poly = poly.transform(test_data)
# 규제,정규화작업 진행
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

train_score = []
test_score = []

# 규제강도 찾기 : 5개 구성
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
# 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_label)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_scaled, train_label))
    test_score.append(lasso.score(test_scaled, test_label))
    
# 선 그래프 
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# Lasso 규제정규화 alpha=10 적용

lasso = Lasso(alpha=10)
lasso.fit(train_scaled,train_label)

print('정확도 : ',lasso.score(train_scaled,train_label))
print('정확도 : ',lasso.score(test_scaled,test_label))