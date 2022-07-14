import sklearn
from tensorflow import keras
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# train : train.csv - label분리
# test : t10k.csv - label 분리, train,test데이터 분리
# 딥러닝을 사용해서 정확도를 출력하시오.

# 데이터 불러오기
train_csv = pd.read_csv('11.deep/d0714/train.csv',header=None)
test_csv = pd.read_csv('11.deep/d0714/t10k.csv',header=None)

# iloc 행으로 가져옴.

# 정규화,표준화 작업
# 함수사용해서 0~1의 값으로 변경
# def func(a):
#     output=[]
#     for i in a:
#         output.append(float(i)/255)
#     return output
# list(map(func,train_csv.iloc[:,1:].values))
    
# 데이터 전처리
# 정규화,표준화작업
train_data = np.array(list(train_csv.iloc[:,1:].values))
train_scaled = train_data/255.0
test_data = np.array(list(test_csv.iloc[:,1:].values))
test_scaled = test_data/255.0

train_label = train_csv[0].values
test_label = test_csv[0].values


# 알고리즘선택 - 다차원배열 사용가능
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(784,)))
model.add(keras.layers.Dense(100,activation='sigmoid'))
model.add(keras.layers.Dense(10,activation='softmax',name="mnist"))
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
# 훈련
model.fit(train_scaled,train_label,epochs=5)

# 딥러닝 model설명
print(model.summary())

# 정확도
score = model.evaluate(test_scaled,test_label)
print(score)