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
# keras의 가장 기본층:Dense층,  10개의 뉴런 생성 - 클래스 10개 같아야 함
# 활성함수 : softmax함수사용 , input_shape(입력층) 개수입력
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')

# 모델객체 생성
model = keras.Sequential([dense1, dense2])


#### summary
model.summary()
# fit(), 미니배치하강법(기본값 32개) 사용 (확률적경사하강법-1개씩, 미니배치-묶음, 배치경사하강법-모든데이터)
# Output Shape에서 (None-기본값을 변동할수 있도록 하기위해 None지정, 100 100개의 뉴런)
# Param 784개입력층 X 은닉층 100개 + 은닉층 100개 = 78500개  784X100+100=78500
# 100개뉴런 X 출력층 10개 + 출력층 10개 = 1010  100x10+100 = 1010
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 100)               78500
#  dense_1 (Dense)             (None, 10)                1010
# =================================================================
# Total params: 79,510
# Trainable params: 79,510
# Non-trainable params: 0

# 모델설정 - 손실함수 적용 : 이진분류:binary_crossetropy, 다중분류:categorical_crossetropy
# 원핫인코딩 1,0,0,0,0,0,0,0,0,0 => 0 그냥숫자로 표기하려면 sparse를 붙임.
# 정확도 측정지표 계산
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
print(train_target[:10])
# 훈련데이터 진행, epochs반복 5회
model.fit(train_scaled, train_target, epochs=5)
# 검증데이터 진행
model.evaluate(val_scaled, val_target)

