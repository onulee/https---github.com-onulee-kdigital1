from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import softmax
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

data = fish[['Weight','Length','Diagonal','Height','Width']]
label = fish['Species']
# numpy변경
# data = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
# label = fish['Species'].to_numpy()

# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 데이터 전처리 스케일 조정, 정규화 작업
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)
test_scaled = ss.transform(test_data)

# 로지스틱 회귀 사용,  확률형태로 구분하기 쉽게 변경
# 다중분류 사용 -> 소프트맥스 함수를 사용해야 함. 
# max_iter는 반복횟수 : 기본값100 -> 1000으로 변경
# max_iter 이 부족하면 부족하다고 에러가 뜸, 크기를 높여주면 됨.
# L2규제 기본적용 - C대문자 규제변수(기본값:1) : 높으면 규제가 약해짐, 낮으면 규제가 높아짐
# 회귀는 alpha규제를 사용하지만 분류에는 C규제변수 사용, C값이 올라가면 규제가 약해짐.
lr = LogisticRegression(C=20,max_iter=1000)
lr.fit(train_scaled,train_label) 
# 정확도 출력
print(lr.score(train_scaled,train_label))  # 0.9327731092436975
print(lr.score(test_scaled,test_label))    # 0.925

print(lr.predict(test_scaled[:5]))

# 클래스 7개(['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']) 정확도 출력
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

# 기울기(가중치), Y절편
print(lr.coef_.shape, lr.intercept_.shape)
# [[0.    0.014 0.841 0.    0.136 0.007 0.003]  <- 3번째 데이터
#  [0.    0.003 0.044 0.    0.007 0.946 0.   ]  <- 6번째 데이터 
print("-"*50)

# 다중분류에서는 로지스틱함수를 사용하면 확률 모든 값이 1이 안됨
# 그래서 소프트맥스 함수를 사용해야 함.
# 소프트맥스 함수 적용 – 다중분류일때 확률형태로 보여지기 위한 함수
# 지수함수를 sum으로 나눠서 모두 더함, 정규화작업, 모두 더하면 결과값이 1이 나옴.
 
# z = -0.404X무게 - 0.576X길이 - 0.663X대각선 - 0.013X높이 - 0.732X두계 - 2.161
# z값을 출력함.
decisions = lr.decision_function(test_scaled[:5])
print(np.round(decisions, decimals=2))

# 양성값만 출력 predict_proba의 1의 값을 출력함.
# [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
# from scipy.special import expit
print(expit(decisions))


