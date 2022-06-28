import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다.
# tf.keras.utils.set_random_seed(42)
# tf.config.experimental.enable_op_determinism()

# train데이터, test데이터 분리
# 훈련데이터 준비
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# train_data 6만개 데이터, 타겟데이터
print(train_input.shape, train_target.shape)
# (60000, 28, 28) (60000,)
# test_data 1만개
print(test_input.shape, test_target.shape)
# (10000, 28, 28) (10000,)


# 이미지 그래프, 출력 
fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    # cmap='gray_r' 이미지를 그레이로 반전출력
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

# 결과값 0-9까지 정수값, 상품이 10개
# [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
print([train_target[i] for i in range(10)])

# return_counts=True로 주면 클래스(결과값) 분포를 볼수 있음
# 0-9까지 총 10개, 몇개가 들어가 있는지 확인가능
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64)) 
print(np.unique(train_target, return_counts=True))

# 로지스틱 회귀 함수 사용
# 이미지데이터 픽셀값이 0-255사이의 데이터 이기에 255로 나눠면 됨.
train_scaled = train_input / 255.0
# 모든데이터를 하나의 배열로 변경 - 2번째,3번째 데이터를 하나로 합침. 
train_scaled = train_scaled.reshape(-1, 28*28)
# 6만개 데이터, 28*28=784
print(train_scaled.shape)
# (60000, 784)

# 사이킷런 1.1.0 버전 이하일 경우 'log_loss'를 'log'로 바꾸어 주세요.
# 경사하강법의 로지스틱회귀 사용, 반복 5번, 
sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)

# 교차 검증
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
