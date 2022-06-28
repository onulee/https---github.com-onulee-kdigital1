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

# 데이터 전처리 스케일 조정, 정규화 작업
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
sc = SGDClassifier(loss='log', random_state=42)
train_score=[]
test_score=[]

# partial_fit메소드에 전달되어질 클래스 개수 전달
classes = np.unique(train_label)
# 300번 정도 반복을 해보고 그 가운데.... 반복횟수를 정함.
for _ in range(0,300):
    # partial_fit메소드는 이전의 가중치,Y절편에서 계속 업데이트하기 위해 사용
    # partial_fit 함수는 데이터의 일부만 전달될수 있다고 가정하기에, 
    # 모두 전달될때도 있지만, 일부만 전달될때도 있기에
    # 클래스(결과값)개수를 꼭 넘겨야 함.
    sc.partial_fit(train_scaled,train_label,classes=classes)
    train_score.append(sc.score(train_data,train_label))
    test_score.append(sc.score(test_data,test_label))

plt.figure(figsize=(8,6)) #그래프 가로세로크기설정
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 30-200 사이의 그래프를 보면서, 반복횟수를 조정해서 최상의 조건을 찾음.    
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled,train_label) 
print("정확도 : ",sc.score(train_scaled,train_label)) 
print("정확도2 : ",sc.score(test_scaled,test_label))




