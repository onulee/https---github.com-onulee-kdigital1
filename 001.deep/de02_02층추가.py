import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


# train데이터, test데이터 분리
# 훈련데이터 준비
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 로지스틱 회귀 함수 사용
# 이미지데이터 픽셀값이 0-255사이의 데이터 이기에 255로 나눠면 됨.
train_scaled = train_input / 255.0
# 모든데이터를 하나의 배열로 변경 - 2번째,3번째 데이터를 하나로 합침. 
train_scaled = train_scaled.reshape(-1, 28*28)
# 6만개 데이터, 28*28=784
print(train_scaled.shape)
# (60000, 784)

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

# -----------------------------------------------------------------------
# Dense층 추가 방법

# 모델객체 생성 - 1. 가장많이 사용, sigmoid함수보다 relu함수를 사용함.
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

# 2. 두번째 사용방법
# model = keras.Sequential([
#     keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
#     keras.layers.Dense(10, activation='softmax', name='output')
# ], name='패션 MNIST 모델')

model.summary()

# 모델설정 - 손실함수 적용 : 이진분류:binary_crossetropy, 다중분류:categorical_crossetropy
# 원핫인코딩 1,0,0,0,0,0,0,0,0,0 => 0 그냥숫자로 표기하려면 sparse를 붙임.
# 정확도 측정지표 계산
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
print(train_target[:10])
# 훈련데이터 진행, epochs반복 5회
model.fit(train_scaled, train_target, epochs=5)
# 검증데이터 진행
print(model.evaluate(val_scaled, val_target))  

