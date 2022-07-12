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

#  가장 낮은 손실값을 저장 - 콜백설정
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, verbose=0,\
    validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

print("stop횟수 : ",early_stopping_cb.stopped_epoch)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

# save파일 읽어오기
model = keras.models.load_model('best-model.h5')
print(model.evaluate(val_scaled, val_target))

