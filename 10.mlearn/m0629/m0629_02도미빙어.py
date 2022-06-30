from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 도미데이터 = 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 
33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 
610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 = 14개 
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 길이 30, 무게 600 -> 도미인지, 빙어인지 분류하시오.


# 1. 데이터 가져오기
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# 1. 길이,무게를 1개의 데이터로 묶음
data = np.column_stack((length,weight))
# 도미:1, 빙어:0
label = np.concatenate((np.ones(35),np.zeros(14)))

# 2.
# data = [[l,w] for l,w in zip(length,weight)]
# data = np.array(data)
# label = [1]*35+[0]*14
# label = np.array(label)

# # 2. 데이터 전처리
train_data,test_data,train_label,test_label = \
    train_test_split(data,label,stratify=label)

# 데이터 전처리 방법2    
# index = np.arange(49)
# np.random.shuffle(index)
# train_data = data[index[:35]]
# test_data = data[index[35:]]
# train_label = label[index[:35]]
# test_label = label[index[35:]]    

# 3. 알고리즘 선택
clf = KNeighborsClassifier()

# 4. 알고리즘 실습훈련
clf.fit(train_data,train_label)

# 5. 예측 - 머신러닝 데이터 2차원배열을 입력을 함.
result = clf.predict([[30,600]])
print("결과값 : ",result)

# 6. 정답률
score1 = clf.score(train_data,train_label)
score2 = clf.score(test_data,test_label)
print("정답률1 : ",score1)
print("정답률2 : ",score2)