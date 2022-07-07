# 2018년 투수의 연봉을 예측하는 알고리즘을 구현하시오
# http://www.statiz.co.kr/
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('10.mlearn/m0707/picher_stats_2017.csv')
# 원핫인코딩 컬럼추가
one_encording = pd.get_dummies(df['팀명'])
df = df.join(one_encording)
print(df)