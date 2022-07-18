from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb

#---------------------------------------------------------
# 파일불러오기
# 웹스크래핑 불러오기
(train_data,train_label),(test_data,test_label) = imdb.load_data(num_words=500)
# (25000,)
print(train_data.shape,test_data.shape)

# 단어의 수 : 218개 글자가 218글자
# 2는 500단어에서 없는 것 표현
# print(train_data[0])
# print(len(train_data[0]))

# 0:부정 1:긍정
# print(train_label)
print(np.unique(train_label))

#---------------------------------------------------------
# 데이터 전처리
sub_data,val_data,sub_label,val_label = train_test_split(train_data,train_label)

# (18750,) (6250,)
print(sub_data.shape,val_data.shape)

# 각 train_data의 문장길이가 어떻게 되는지 확인
# 25000개의 단어길이를 합을 구함.
lengths = np.array([len(x) for x in train_data])

# 평균값:238.71364  중간값 : 178.0 최대값 :2494  최소값:11
print(np.mean(lengths),np.median(lengths))

# 그래프그리기
plt.hist(lengths)
plt.xlabel('lengths')
plt.ylabel('frequency')
plt.show()


# 11~2494 글자 -> 100글자만 사용
# 100글자 짜르고, 없는 부분 0으로 채워줌
from tensorflow.keras.preprocessing.sequence import pad_sequnces

train_seq = pad_sequnces(sub_data,maxlen=100)
# 문장길이 218 -> 문장길이 100으로 변경
print(train_seq[0])

