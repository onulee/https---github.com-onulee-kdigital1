from random import weibullvariate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
print('bream_length : ',len(bream_length))

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
print('smelt_length : ',len(smelt_length))

# 25,150추가
# 도미, 빙어 산점도그래프 그리기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(25,150)
plt.xlabel('길이')
plt.ylabel('무게')
plt.xticks(range(0,1200,200))
plt.show()

# 도미, 빙어 데이터 합치기
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# numpy배열을 사용하여 train,test데이터 분리
# 데이터를 열기준으로 세로로 정렬후 2개의 행을 묶음
data = np.column_stack((length,weight))
# 2개의 배열을 묶음
label = np.concatenate((np.ones(35),np.zeros(14)))

# ---------------------------------------------
# 1. numpy를 이용한 train,test데이터 분리
data_numpy = np.array(data) 
label_numpy = np.array(label) 
index = np.arange(49)
np.random.shuffle(index)

train_data = data_numpy[index[:35]]
test_data = data_numpy[index[35:]]
train_label = label_numpy[index[:35]]
test_label = label_numpy[index[35:]]
# ---------------------------------------------

# ---------------------------------------------
# 2. sklearn 라이브러리에서 랜덤으로 실습데이터,테스트데이터 분리
# train_data,test_data,train_label,test_label = train_test_split(data,label)
# ---------------------------------------------

# k근접알고니즘
clf = KNeighborsClassifier()
# clf = svm.SVC()
clf.fit(train_data,train_label) 
result =clf.predict(test_data)
result2 =clf.predict([[25,150]])   #결과값 : 빙어로 나옴. 실제는 도미가 맞음.
print('결과값 : ',result2)  # 0-setosa

score = clf.score(test_data,test_label)
print('정답률 : ',score) 