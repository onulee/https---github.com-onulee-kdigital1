import sklearn
from tensorflow import keras
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# mnist파일 불러오기
# keras mnist파일이 설치될때 자동설치가 됨.
(train_data,train_label),(test_data,test_label) = keras.datasets.fashion_mnist.load_data()
# train_data,test_data,train_label,test_label = train_test_split()

print(train_data.shape, train_label.shape)
print(test_data.shape,test_label.shape)

print(train_label[:10])

# 이미지 출력
fig,axs = plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    # imshow: 그레이출력, gray_r : 반전
    axs[i].imshow(train_data[i],cmap='gray_r')
    axs[i].axis('off')
plt.show()    

# 전처리

# 알고리즘 선택

# 실습훈련

# 정확도