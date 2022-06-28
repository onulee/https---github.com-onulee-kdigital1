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
# Flatten input 28*28 이므로 그대로 대입
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
# relu활성함수 -> sigmoid함수 대체 (단점 : 선형함수가 더커지거나, 작아지면 변화가 없음.)
# relu함수는 0보다 클때는 그대로 적용, 0이하는 0으로 대체
# flatten층 의미가 없음, 단 28*28을 1차원으로 자동으로 변경해서 전달함.
# train_scaled.reshape(-1, 28*28) 사용할 필요 없음.
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


model.summary()

# 모델설정 - 손실함수 적용 : 이진분류:binary_crossetropy, 다중분류:categorical_crossetropy
# 원핫인코딩 1,0,0,0,0,0,0,0,0,0 => 0 그냥숫자로 표기하려면 sparse를 붙임.
# 정확도 측정지표 계산
# ------------------------------------------------------------------------
# keras.optimizers.SGD(learning_rate 기본값은 0.01 -> 변경하고 싶을때 선언후 사용해야 함.

# 1.SGD
# sgd = keras.optimizers.SGD(learning_rate=0.1)
# model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.33834901452064514, 0.875]

# 2.momentum
# momentum = keras.optimizers.SGD(momentum=0.9, nesterov=True)
# model.compile(optimizer=momentum, loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.3613305985927582, 0.8675000071525574]

# 3. adagrad
# adagrad = keras.optimizers.Adagrad()
# model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.6081475019454956, 0.7988333106040955]

# 4. adam - 가장 좋은 정확도가 나타남.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.32611122727394104, 0.8831666707992554]

#  # 옵티마이저 : 기본형태
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
# sgd = keras.optimizers.SGD()
# model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')

print(train_target[:10])
# 훈련데이터 진행, epochs반복 5회
model.fit(train_scaled, train_target, epochs=5)
# 검증데이터 진행
print(model.evaluate(val_scaled, val_target))  

