import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus']=False

import pandas as pd
import numpy as np
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
print('bream_length : ',len(bream_length))

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
print('smelt_length : ',len(smelt_length))


#----------------------------------------------
# 도미, 빙어 산점도그래프 그리기
# 30,600추가
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30,600, marker='^')  # 30,600추가
plt.xlabel('길이')
plt.ylabel('무게')
plt.show()

# 도미, 빙어 데이터 합치기
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

# 2개의 데이터를 1개의 형태로 묶음 - 앞쪽35, 뒤쪽14
data = [[l, w] for l, w in zip(length, weight)]
print(data)
target = [1]*35 + [0]*14
 
# k근접알고니즘
clf = KNeighborsClassifier()
# clf = svm.SVC()
clf.fit(data,target) 
result =clf.predict(data)
result2 =clf.predict([[30,600]])
print('결과값 : ',result)  # 0-setosa
print('결과값2 : ',result2)  # 0-setosa

score = clf.score(data,target)
print('정답률 : ',score) 