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


# 딥러닝알고리즘
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

# 1. 기본옵티마이저 SGD확률적경사하강법
# model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',\
#     metrics='accuracy')

# 2. 옵티마이저 분리 SGD확률적경사하강법 learning_rate=0.01
# s = keras.optimizers.SGD(learning_rate=0.1)
# model.compile(optimizer=s,loss='sparse_categorical_crossentropy',\
#     metrics='accuracy')

# 3. momentum
# momentum = keras.optimizers.SGD(momentum=0.9,nesterov=True)
# model.compile(optimizer=momentum,loss='sparse_categorical_crossentropy',\
#     metrics='accuracy')

# 4. adam
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',\
    metrics='accuracy')


#---------------------------------------------
model.summary()
model.fit(train_scaled,train_label)
score = model.evaluate(test_scaled,test_label)


