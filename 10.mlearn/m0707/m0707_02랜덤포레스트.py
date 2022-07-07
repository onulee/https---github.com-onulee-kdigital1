from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier # 확률적경사하강법
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split # train,test
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler     # 정규화,표준화작업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 검증세트추가
wine = pd.read_csv('10.mlearn/m0706/wine.csv')
# print(wine.info())
data = wine[['alcohol','sugar','pH']].to_numpy()
label = wine['class'].to_numpy()

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(
    data,label, random_state=42)

# 알고리즘 선택
# cpu core개수를 몇개를 사용할지 정함. -1 core를 사용함.
# 랜덤포레스트 : 결정트리사용 디폴트 10개사용, n_estimators=100
rf = RandomForestClassifier(n_jobs=-1,random_state=42)

# 교차검증훈련
# return_train_score = True : train_score가 출력
scores = cross_validate(rf,train_data,train_label,return_train_score=True,n_jobs=-1)

print("train_score 평균 : ",np.mean(scores['train_score']))
print("test_score 평균 : ",np.mean(scores['test_score']))

# 정확도
rf.fit(train_data,train_label)
print("test_data score : ",rf.score(test_data,test_label))

# 특성의 중요도
print(rf.feature_importances_)

