import sklearn
from tensorflow import keras
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# mnist파일 불러오기
# keras mnist파일이 설치될때 자동설치가 됨.
(train_data,train_label),(test_data,test_label) = keras.datasets.fashion_mnist.load_data()
# train_data,test_data,train_label,test_label = train_test_split()

print(train_data.shape, train_label.shape)
print(test_data.shape,test_label.shape)

print(train_label[:10])

# 이미지 출력
fig,axs = plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    # imshow: 그레이출력, gray_r : 반전
    axs[i].imshow(train_data[i],cmap='gray_r')
    axs[i].axis('off')
plt.show()    

# ----------------------------------------------------------
# train데이터, test데이터 분리 : test_size=0.2 , test데이터 20%
train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

# train_data 6만개 데이터, 타겟데이터
print(train_scaled.shape, train_target.shape)
# (48000, 784) (48000,)
# test_data 1만개
print(val_scaled.shape, val_target.shape)
# (12000, 784) (12000,)

# keras의 가장 기본층:Dense층,  10개의 뉴런 생성 - 클래스 10개 같아야 함
# 활성함수 : 다중분류 softmax함수사용 , input_shape(입력층) 개수입력
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# 모델객체 생성
model = keras.Sequential(dense)

# 다른 방법1 - Sequential()안에 Dense넣기
# model = keras.Sequential(keras.layers.Dense(10, activation='softmax', input_shape=(784,)))

# 다른 방법2
# model = keras.Sequential() - add로 Dense넣기
# model.add(keras.layers.Flatten(input_shape=(784,)))
# model.add(keras.layers.Dense(10, activation='softmax'))

# 모델설정 - 손실함수 적용 : 이진분류:binary_crossetropy, 다중분류:categorical_crossetropy
# 원핫인코딩 1,0,0,0,0,0,0,0,0,0 => 0 그냥숫자로 표기하려면 sparse를 붙임.
# 정확도 측정지표 계산
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
print(train_target[:10])
# 훈련데이터 진행, epochs반복 5회
model.fit(train_scaled, train_target, epochs=5)
# 검증데이터 진행
print(model.evaluate(val_scaled, val_target))