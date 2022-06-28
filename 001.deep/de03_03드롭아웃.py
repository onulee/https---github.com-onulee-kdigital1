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
# Dropout추가 30%로 은닉층의 유런을 숨김
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# --------------------------------------------------------------------------
# 손실곡선 그리기
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

#---------------------------------------------------------------------------
# validation_data=(val_scaled, val_target) 매개변수
# 검증데이터로 테스트를 하면, 각 epoch마다 훈련손실, 정확도 출력
 
# train데이터에서는 정확도가 올라가지만, test데이터에서는 정확도가 내려갈수 있어서 
# 2개의 그래프를 출력해서 확인을 해야 함.
history = model.fit(train_scaled, train_target, epochs=20, verbose=0,validation_data=(val_scaled,val_target))
# loss,accuracy 출력
print(history.history.keys())
# 4개값 : dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 손실곡선 그래프 출력, loss출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 그래프를 보면 epoch가 5~6을 넘어가면 손실이 높아짐.



