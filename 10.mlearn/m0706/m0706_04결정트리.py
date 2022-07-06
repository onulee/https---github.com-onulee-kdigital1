from sklearn.linear_model import LogisticRegression, SGDClassifier # 확률적경사하강법
from sklearn.model_selection import train_test_split # train,test
from sklearn.preprocessing import PolynomialFeatures, StandardScaler     # 정규화,표준화작업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 데이터불러오기 0:red wine, 1:white wine
# 특성 : alcohol,sugar,pH  클래스 : 2개 class
# wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine = pd.read_csv('10.mlearn/m0706/wine.csv')
print(wine.info())
data = wine[['alcohol','sugar','pH']].to_numpy()
label = wine['class'].to_numpy()

# 데이터 전처리
train_data, test_data, train_label, test_label = train_test_split(
    data, label, random_state=42)

print(train_data.shape)

# 정규화,표준화작업
# ss = StandardScaler()
# train_scaled = ss.fit_transform(train_data,train_label) 
# test_scaled = ss.fit_transform(test_data,test_label) 

# for문
# train_score=[]
# test_score=[]

# for idx in range(1,20):
#     dt = DecisionTreeClassifier(max_depth=idx, random_state=42)
#     dt.fit(train_data,train_label)
#     train_score.append(dt.score(train_data,train_label))
#     test_score.append(dt.score(test_data,test_label))

# plt.plot(range(1,20),train_score)
# plt.plot(range(1,20),test_score)
# plt.show()


# 알고리즘 선택 - 결정트리 분류, max_depth
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# 훈련
dt.fit(train_data,train_label)

# 특성중요도 - 특성 어떤게 가장 영향을 미치는지 확인
print(dt.feature_importances_)
# 예측
# 정확도
print("train score : ",dt.score(train_data,train_label))
print("test score : ", dt.score(test_data,test_label))

# 그래프 그리기
# 사이즈 확대
plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=1,filled=True,feature_names=['alcohol','sugar','pH'])
# plot_tree(dt,max_depth=1)
plt.show()