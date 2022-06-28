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

# 4. 정규화 작업
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 1차적으로 로지스틱 함수를 사용해서 진행
# # 5. 로지스틱 알고니즘 적용
# lr = LogisticRegression()
# lr.fit(train_scaled, train_target)

# print(lr.score(train_scaled, train_target))
# print(lr.score(test_scaled, test_target))

# # 기울기(가중치), Y절편
# print(lr.coef_, lr.intercept_)

# 5. 트리알고니즘 적용
# max_depth=3, depth를 3까지만 가능
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled,train_target)

# 그래프 그리기
plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=1, filled=True, feature_names=['alcohol','sugar','pH'])
plt.show()

# 특성 중요도 - 특성중에 어떤것이 가장 영향이 큰지를 알수 있음.
print(dt.feature_importances_)

print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))
