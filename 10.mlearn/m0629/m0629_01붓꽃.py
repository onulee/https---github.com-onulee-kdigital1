# iris(150).csv 
# 특정데이터를 입력하여서 붓꽃 데이터의 품종을 
# 분류하는 프로그램을 구축하시오.
# SepalLength,SepalWidth,PetalLength,PetalWidth,Species
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. 데이터를 가져오기
df = pd.read_csv('10.mlearn/m0629/iris(150).csv')
# print(df)

# 2. 데이터전처리
# train_data,train_label,test_data,test_label
data =df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
label = df[['Species']]
train_data,test_data,train_label,test_label = \
    train_test_split(data,label,stratify=label)

# numpy 타입변경    
# data_numpy = np.array(data)
# index = np.arange(150)
# np.random.shuffle(index)
# train_data = data_numpy[index[:120]]    
    

# 3. 알고리즘선택 : knn

clf = KNeighborsClassifier()
# 4. 알고리즘 실습훈련
clf.n_neighbors=4
clf.fit(train_data,train_label)

# 5. 예측
result = clf.predict(test_data)
print("결과값 : ",result)

# 6. 정답률
score1 = clf.score(train_data,train_label)
score2 = clf.score(test_data,test_label)
print('정답률1 : ',score1)
print('정답률2 : ',score2)