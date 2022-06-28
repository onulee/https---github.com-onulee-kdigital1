from turtle import distance
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 도미데이터 = 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 
33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 
610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 = 14개 1
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
length = bream_length + smelt_length 
weight = bream_weight + smelt_weight

# 1개의 list합치기
# 1. 데이터 전처리
# numpy 2개의 list를 1개의 array생성
data = np.column_stack((length,weight))
# ones,zeros array를 1개의 array로 변형
label = np.concatenate((np.ones(35),np.zeros(14)))

# 1) train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label)

# 정규화작업 2
ss = StandardScaler()
ss.fit(train_data)   
train_scaled = ss.transform(train_data) #(data-산술평균)/표준편차
test_scaled = ss.transform(test_data) #(data-산술평균)/표준편차
new = ss.transform([[25,150]])

# 정규화 작업 1
# mean = np.mean(data,axis=0) # 산술평균
# std = np.std(data,axis=0)   # 표준편차
# train_scaled = (data-mean)/std
# new = ([25,150] - mean)/std

# 2. 알고리즘 선택
clf = KNeighborsClassifier()

# 3. 데이터 학습훈련
clf.fit(train_scaled,train_label)
# clf에 있는 train_data의 knn이웃하는 5개의 데이터를 추출

# 4. 데이터 예측 (길이:30, 무게 600)
result = clf.predict(new)
print('결과값2 : ',result)

# 5. 정답률
score = clf.score(test_scaled,test_label)
print("정답률 : ",score)

# 그래프 작업
plt.scatter(train_scaled[:,0],train_scaled[:,1])
# plt.scatter(new[0],new[1], marker='x')
plt.show()






