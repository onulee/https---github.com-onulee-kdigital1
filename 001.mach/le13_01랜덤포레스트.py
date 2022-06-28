from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 가져오기
wine = pd.read_csv('https://bit.ly/wine_csv_data')
print(wine.head())
# print(wine['alcohol'].mean())
# print(wine.columns)
print(wine.info())     # null공백이 없는지 확인
print(wine.describe()) # 전반적인 정보확인

# 2. data, label분리
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 3. train,test데이터 분리
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)


# 4. 랜덤포레스트 알고리즘 사용, 모든score 사용 : n_jobs=-1
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
# default:test_score만, train_score까지 출력
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
# test데이터는 따로 있음.
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9973541965122431 0.8905151032797809

rf.fit(train_input, train_target)
print(rf.feature_importances_)
# 특성값을 고르게 분포
# [0.23167441 0.50039841 0.26792718]

# oob : 랜덤으로 샘플을 사용하고 남는 샘플을 다시 train을 함.
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
# 남는 샘플을 가지고 train 결과값
print(rf.oob_score_)
# 0.8934000384837406

