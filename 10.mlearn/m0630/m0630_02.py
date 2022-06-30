from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 도미데이터 = 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 
33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 
610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 = 14개 1
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]




# 1. 데이터 전처리 ( 특성, 분류 )
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 특성,분류 -> 숫자의 범위가 넓으면, 표준점수로 변환
data = np.column_stack((length,weight))
# label = '도미'*35+'빙어'*14
label = np.concatenate((np.zeros(35),np.ones(14)))

# train데이터, test데이터 분리
train_data,test_data,train_label,test_label = train_test_split(data,label)
    
# 표준점수 변환 -> 정규화작업, 규제
ss = StandardScaler()
ss.fit(train_data)
train_scaled = ss.transform(train_data)  # (train_data-산술평균)/표준편차
test_scaled = ss.transform(test_data)    # (test_data-산술평균)/표준편차
new = ss.transform([[25,150]])

# 2. 알고리즘 선택 - 
clf = KNeighborsClassifier()

# 3. 실습훈련
clf.fit(train_data,train_label)

# 4. 예측
result = clf.predict(test_data)
print("결과값 : ",result)
result2 = clf.predict([[25,150]])
print("25,150 결과값 0-도미,1-빙어 : ",result2)


# 5. 정답률   
score = clf.score(test_scaled,test_label)
print("정답률 : ",score)
