from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1.데이터 불러오기 : keras mnist파일을 불러오기
# info, describe
# shape : (60000,28,28)
# class : 0-9까지 10개
(train_data,train_label),(test_data,test_label) = keras.datasets.fashion_mnist.load_data()

# print(train_data.shape)
# print(train_label[:5])
# print(np.unique(train_label))

# 2. 정규화, 표준화작업
train_data = train_data/255
test_data = test_data/255

# 3. train,test데이터 분리
# (45000, 28, 28)
# (15000, 28, 28)
train_scaled,val_scaled,train_label,val_label = train_test_split(train_data,train_label)

# ------------------------------------------------------
# model 불러오기
# ------------------------------------------------------
# 4. 딥러닝선언 - 인공신경망 ANN,DNN  / 합성곱신경망-CNN  / 순환신경망 - RNN
# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(28,28)))
# model.add(keras.layers.Dense(100,activation='relu'))
# model.add(keras.layers.Dense(10,activation='softmax'))
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

# # save_weights파일 읽어오기
# model.load_weights('model_test.h5')

# save model전체저장 파일 불러오기
# model = keras.models.load_model('best-model.h5')
model = keras.models.load_model('model-all.h5')

# argmax : 최대값의 주소 반환
# axis 방향 -> 열의 방향 val_scaled 12000개
val_labels = np.argmax(model.predict(val_scaled),axis=-1)
# val_labels: 주소값 -> 데이터값을 검색해야 함.
print(np.mean(val_labels==val_label))


# 6. 정확도
score = model.evaluate(val_scaled,val_label)
print(score)


