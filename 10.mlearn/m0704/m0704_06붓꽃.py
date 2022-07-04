from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터불러오기,데이터분할 : pandas - read_csv,to_csv,to_numpy
# 데이터의 배열형태 변경 : numpy - array,column_stack,concatenate
df = pd.read_csv('10.mlearn/m0628/iris(150).csv')

# data 2차원 numpy배열
data = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].to_numpy()
label = df['Species'].to_numpy()


# 1. 데이터전처리 - train_data,test_data분리
train_data,test_data,train_label,test_label = train_test_split(data,label)

# 2. 알고리즘 선택
clf = KNeighborsClassifier()

# 3. 실습훈련 - fit
clf.fit(train_data,train_label)

# 4. 예측 - predict : 5.2 3.4 4.8 0.2
result = clf.predict([[5.2,3.4,4.8,0.2]])
print("예측결과(5.2,3.4,4.8,0.2) : ",result)

# 5. 정확도(정답률)-score
score = clf.score(test_data,test_label)
print("정답률 : ",score)