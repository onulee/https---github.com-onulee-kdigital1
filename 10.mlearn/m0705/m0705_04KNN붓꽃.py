from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit, softmax             # z점수 0-1사이의 값으로 변경
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('10.mlearn/m0705/iris(150).csv',index_col='caseno')
data = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].to_numpy()
label = df['Species'].to_numpy()

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label)

# 정규화,표준화작업 -  5.1 3.8 1.8  0.5
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)
new_scaled = ss.fit_transform([[5.1,3.8,1.8,0.5]])

# 알고리즘 선택
clf = KNeighborsClassifier(n_neighbors=7)

# 훈련
clf.fit(train_scaled,train_label)

# 예측
result = clf.predict(new_scaled)
result2 = clf.predict_proba(new_scaled)
print("예측결과 : ",result)
print("예측결과proba : ",result2)

# 정확도
score1 = clf.score(train_scaled,train_label)
score2 = clf.score(test_scaled,test_label)
print("정확도 train : ",score1)
print("정확도 test : ",score2)

