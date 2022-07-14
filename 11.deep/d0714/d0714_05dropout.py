import sklearn
from sklearn import metrics
from tensorflow import keras
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# mnist파일 불러오기
# keras mnist파일이 설치될때 자동설치가 됨.
(train_data,train_label),(test_data,test_label) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_data/255.0
test_scaled = test_data/255.0

sub_scaled,val_scaled,sub_label,val_label=train_test_split(train_scaled,train_label)


# 딥러닝 선언
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation='relu'))
# 규제를 통한 train정확도를 낮춤.
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

# 설정
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy'\
    ,metrics='accuracy')

# 훈련
history = model.fit(sub_scaled,sub_label,epochs=20,\
    validation_data=(val_scaled,val_label))

# 그래프
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['sub','val'])
plt.show()