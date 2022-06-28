from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import numpy as np
import pandas as pd

# 데이터 생성 - Species,Weight,Length,Diagonal,Height,Width
fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish.head())
# Species 중복제거 출력
print(pd.unique(fish['Species']))

# data = fish[['Weight','Length','Diagonal','Height','Width']]
# label = fish['Species']
# numpy변경
data = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
label = fish['Species'].to_numpy()

# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 데이터 전처리 스케일 조정, 정규화 작업
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)
test_scaled = ss.transform(test_data)

# 이진분류일때 구성을 먼저 알아봄. Bream,Smelt는 True, 그외 False
# 결과값을 2개로 한정 : Bream,Smelt <- 'Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt'
bream_smelt_index = (train_label == 'Bream') | (train_label =='Smelt')
print(bream_smelt_index)
train_bream_smelt = train_scaled[bream_smelt_index]
label_bream_smelt = train_label[bream_smelt_index]

# 로지스틱 회귀 사용,  확률형태로 구분하기 쉽게 변경
lr = LogisticRegression()
lr.fit(train_bream_smelt,label_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

# 기울기(가중치), Y절편
print(lr.coef_,lr.intercept_)
# [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
print("-"*50)

# z = -0.404X무게 - 0.576X길이 - 0.663X대각선 - 0.013X높이 - 0.732X두계 - 2.161
# decision_function : z값을 출력함.
decisions = lr.decision_function(test_scaled[:5])
print(np.round(decisions, decimals=2))
# [ 0.92  3.19 -1.05 -1.53  0.08]

# 양성값만 출력 predict_proba의 1의 값을 출력함.
# from scipy.special import expit : 시그모이드 함수 출력
print(expit(decisions))

# 1의 값과 같음.
# 0.00240145 = 0.00240145
# [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
# [[0.99759855 0.00240145]
#  [0.02735183 0.97264817]
#  [0.99486072 0.00513928]
#  [0.98584202 0.01415798]
#  [0.99767269 0.00232731]]



