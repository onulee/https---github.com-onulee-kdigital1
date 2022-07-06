from sklearn.linear_model import LogisticRegression, SGDClassifier # 확률적경사하강법
from sklearn.model_selection import train_test_split # train,test
from sklearn.preprocessing import PolynomialFeatures, StandardScaler     # 정규화,표준화작업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 확률적경사하강법, 데이터불러오기 0:red wine, 1:white wine
# 특성 : alcohol,sugar,pH  클래스 : 2개 class
wine = pd.read_csv('10.mlearn/m0706/wine.csv')
print(wine.info())
data = wine[['alcohol','sugar','pH']].to_numpy()
label = wine['class'].to_numpy()

# 로지스틱 회귀 - 정확도를 출력하시오.
# 데이터 전처리
train_data, test_data, train_label, test_label = train_test_split(
    data, label, random_state=42)
# x_data, x_data, y_label, y_label = train_test_split(
#     x, y, random_state=42)

# 정규화,표준화작업
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)

# SGDClassifier
# sc = SGDClassifier(loss='log', random_state=42)
# train_score=[]
# test_score=[]
# classes = np.unique(train_label)
# for _ in range(0,300):
#    sc.partial_fit(train_scaled,train_label,classes=classes)
#    train_score.append(sc.score(train_scaled,train_label))
#    test_score.append(sc.score(test_scaled,test_label)) 
   
# plt.figure(figsize=(8,6)) #그래프 가로세로크기설정
# plt.plot(train_score)
# plt.plot(test_score)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()   


sc = SGDClassifier(loss='log_loss', max_iter=250, tol=None, random_state=42)
sc.fit(train_scaled, train_label)
print(sc.score(train_scaled, train_label))
print(sc.score(test_scaled, test_label))
 
   