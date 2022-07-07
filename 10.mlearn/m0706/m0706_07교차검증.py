from sklearn.linear_model import LogisticRegression, SGDClassifier # 확률적경사하강법
from sklearn.model_selection import cross_validate, train_test_split # train,test
from sklearn.preprocessing import PolynomialFeatures, StandardScaler     # 정규화,표준화작업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 검증세트추가
wine = pd.read_csv('10.mlearn/m0706/wine.csv')
print(wine.info())
data = wine[['alcohol','sugar','pH']].to_numpy()
label = wine['class'].to_numpy()

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(
    data,label, random_state=42)

# 알고리즘 선택 
dt = DecisionTreeClassifier(random_state=42)

# 훈련 : 교차검증함수 cross_validate(알고리즘,data,label)
# fit_time:훈련시간,score_time:model score값 출력시간,test_score: score정확도
scores = cross_validate(dt,train_data,train_label)
# print(scores)

# 정확도
print("train score : ",np.mean(scores['test_score']))

dt.fit(train_data,train_label)
# test_data
print("test score : ",dt.score(test_data,test_label))
