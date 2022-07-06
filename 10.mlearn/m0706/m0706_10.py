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

# min_impurity_decrease 옵션 증가
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }

# 그리드서치적용 - 알고리즘 선택 
# n_jobs=-1 , cpu core 모든core 사용함.
gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)

# 훈련 - 5번을 반복훈련
gs.fit(train_data,train_label)

# 훈련후 가장 성능이 우수한 fit을 dt에 넣어줌.
dt = gs.best_estimator_

# 훈련후 가장 성능이 우수한 params의 값을 추출 
print(gs.best_params_) 

# 정확도
print("train score : ",dt.score(train_data,train_label))

# 평균score
print(gs.cv_results_['mean_test_score'])