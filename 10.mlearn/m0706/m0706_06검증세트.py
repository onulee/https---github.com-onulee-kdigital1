from sklearn.linear_model import LogisticRegression, SGDClassifier # 확률적경사하강법
from sklearn.model_selection import train_test_split # train,test
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

# 데이터 전처리 - 최종 런칭할때 마지막으로 테스트를 하는 용도로 사용함.
train_data, test_data, train_label, test_label = train_test_split(
    data, label, random_state=42)

# 검증세트 추가
sub_data,val_data,sub_label,val_label = train_test_split(
  train_data,train_label,random_state=42)


# 알고리즘 선택 - 결정트리 분류, max_depth
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# 훈련
dt.fit(sub_data,sub_label)

# 특성중요도 - 특성 어떤게 가장 영향을 미치는지 확인
print(dt.feature_importances_)
# 예측
# 정확도
print("train score : ",dt.score(sub_data,sub_label))
print("test score : ", dt.score(val_data,val_label))

# 그래프 그리기
# 사이즈 확대
plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=1,filled=True,feature_names=['alcohol','sugar','pH'])
# plot_tree(dt,max_depth=1)
plt.show()