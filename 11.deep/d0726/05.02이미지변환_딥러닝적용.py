import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

# 카테고리 지정하기
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64 
image_h = 64

# 데이터 불러오기 --- (※1)
X_train, X_test, y_train, y_test = np.load("11.deep/d0726/image/5obj.npy")
# 데이터 정규화하기
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float")  / 255
print('X_train shape:', X_train.shape)

# 딥러닝 선언
# 합성곱 신경망 선언
model = keras.Sequential()

# CNN
model.add(keras.layers.Conv2D(32,kernel_size=3,activation='relu',padding='same',input_shape=(28,28,1)))
# 최대풀링
model.add(keras.layers.MaxPooling2D(2))

# CNN - 1회 반복
model.add(keras.layers.Conv2D(64,kernel_size=3,activation='relu',padding='same'))
# 최대풀링
model.add(keras.layers.MaxPooling2D(2))
# 딥러닝 훈련

# 딥러닝 평가




