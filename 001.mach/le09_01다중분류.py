from sklearn import svm
from sklearn.model_selection import train_test_split
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
# fish.to_csv('fish_data.csv',encoding='utf-8-sig',index=False)
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

# knn알고니즘 대입
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train_scaled,train_label) 

print("결과값 항목 : ",clf.classes_) 
#특성,결과값 항목들이 어떤것이 있는지 모두 보여줌.
# 결과값 항목 :  ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish'] 알파벳 순으로 정렬

# 5개만 우선 확인
result =clf.predict(test_scaled[:5])
print("결과값 : ",result)
# 결과값 :  ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']

# 특성별로 결과값 확률이 나타남
proba = clf.predict_proba(test_scaled[:5]) 
print(np.round(proba, decimals=4))
# 특성,결과값을 소수점 4자리로 통일
# .... 
#  [0.     0.     0.     1.     0.     0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
# 문제는 n_neighbors=3 이기에 확률은 0,1/3, 2/3, 3/3 만 나옴.
# 이러한 부분을 해결하기 위해 로지스틱 회귀를 사용

score = clf.score(train_scaled,train_label)
score2 = clf.score(test_scaled,test_label)
print('정답률 : ',score) 
print('정답률2 : ',score2) 
