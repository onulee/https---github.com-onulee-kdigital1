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

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# -----------------------------------------------------------------------
# 모델객체 생성 - 1. 가장많이 사용, sigmoid함수보다 relu함수를 사용함.
# Dropout추가 30%로 은닉층의 유런을 숨김
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
# --------------------------------------------------------------------------
# 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

history = model.fit(train_scaled, train_target, epochs=20, verbose=0,validation_data=(val_scaled,val_target))

# 파이썬 객체로 모델을 저장 후 불러오기
# keras에서는 2가지 : save_weights - 가중치,절편만 저장(모듈파라미터만 저장),모듈구조는 저장하지 않음
#                                 - model을 함수로 생성해서 저장해야 함.
# save함수 - 모델과 가중치를 모두 저장, model을 함수로 만들 필요가 없음.
model.save_weights('model-weights.h5')

model.save('model-whole.h5')

# save_weights파일 읽어오기
model.load_weights('model-weights.h5')

# np.argmax : 각 샘플마다 가장 큰 정확도를 찾음. 
# axis = -1 은 axis = 1과 같음.
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
# val_labels와 val_target값이 같은 경우만 평균
print(np.mean(val_labels == val_target))

# save파일 읽어오기
model = keras.models.load_model('model-whole.h5')

model.evaluate(val_scaled, val_target)

