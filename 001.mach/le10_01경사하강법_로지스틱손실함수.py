from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
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

# numpy변경
data = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
label = fish['Species'].to_numpy()

# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 경사하강법은 반드시 데이터 전처리를 해줘야 함.
# 데이터 전처리 스케일 조정, 정규화 작업
# 훈련세트와 테스트 세트 모두 적용해야 함.
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)
test_scaled = ss.transform(test_data)


# SGDClassifier:분류모델일 경우 -> 반대 SGDRegressor:회귀모델일 경우
# SGDClassifier모델알고니즘이 아니라 방법에 속하기에 어떤 함수를 적용할지를 지정
# loss='log' 로지스틱 손실함수 적용
# 결과를 동일하게 하기 위해 random_state=42
# max_iter는 반복횟수
# SDGClassifier는 확률적 경사하강법만 가능, 배치,미니배치는 지원하지 않음
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled,train_label) 
print("정확도 : ",sc.score(train_scaled,train_label))
print("정확도2 : ",sc.score(test_scaled,test_label))

# fit 함수를 사용하면 기울기(가중치),Y절편을 모두 버리고 다시 설정함.
# partial_fit은 기존의 기울기(가중치).Y절편을 가지고 다시 훈련을 진행
sc.partial_fit(train_scaled,train_label)

print("정확도 : ",sc.score(train_scaled,train_label)) 
print("정확도2 : ",sc.score(test_scaled,test_label))




