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
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# ------------------------------------------------------------------------
# optimizers 는 기본 하이퍼파라미터여서, for문으로 변경을 하면서 최적의 형태를 적용
# 1.옵티마이저 : 기본형태 SGD - compile안에 optimizer를 포함
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')

# 2.SGD - optimizers를 분리해서 진행 : learning_rate를 설정할수 있음
# sgd = keras.optimizers.SGD(learning_rate=0.1)
# model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.33834901452064514, 0.875]

# 3.momentum
# momentum = keras.optimizers.SGD(momentum=0.9, nesterov=True)
# model.compile(optimizer=momentum, loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.3613305985927582, 0.8675000071525574]

# 4. adagrad - 적응적 학습률 옵티마이저
# adagrad = keras.optimizers.Adagrad()
# model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.6081475019454956, 0.7988333106040955]

# 5. adam - 적응적 학습률 옵티마이저 : 가장 좋은 정확도가 나타남.
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# [0.32611122727394104, 0.8831666707992554]

print(train_target[:10])
# 훈련데이터 진행, epochs반복 5회
model.fit(train_scaled, train_target, epochs=5)
# 검증데이터 진행
print(model.evaluate(val_scaled, val_target))  

