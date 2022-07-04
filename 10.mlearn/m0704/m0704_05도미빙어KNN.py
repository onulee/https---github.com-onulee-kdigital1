import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #KNN알고리즘
from sklearn.svm import SVC  # SVC알고리즘


# 데이터를 훈련시켜서, 도미와 빙어를 분류하는 알고리즘 구현하시오.
# 물고기의 길이,무게를 넣으면 도미, 빙어입니다.

# 도미데이터 = 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 
33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 
610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 = 14개 
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


# 데이터 만들어야 함. - train_data,test_data 원본 2차원 데이터
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# 머신러닝에 넣을 데이터는 2차원배열이어야 함. 기본타입 numpy배열
data = np.column_stack((length,weight)) # (49, 2)
# label = np.concatenate((np.ones(35),np.zeros(14))) # 1차원배열
label = np.concatenate((np.array(['도미']*35),np.array(['빙어']*14))) # 1차원배열


# 1. 데이터 전처리,분리
# train_data,test_data 분리하는 이유 : 훈련하는 데이터를 시험을 볼수 없음.
# train_data,test_data자동분리
train_data,test_data,train_label,test_label = train_test_split(data,label)

# 2. 알고리즘 선택
clf = KNeighborsClassifier()
# clf = SVC()

# 3. 실습훈련-fit 36개
clf.fit(train_data,train_label)

# 4. 예측-predict, 도미,빙어인지 분류
result = clf.predict([[30,600]])
print("30cm,600g은 무슨 물고기일까요? ",result)

# 5. 정확도-score
score = clf.score(test_data,test_label)
print("정확도(정답률) :",score)



