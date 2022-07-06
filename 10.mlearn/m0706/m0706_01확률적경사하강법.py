from sklearn.linear_model import SGDClassifier # 확률적경사하강법
from sklearn.model_selection import train_test_split # train,test
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 데이터불러오기
fish_df = pd.read_csv('10.mlearn/m0706/fish.csv')
data=fish_df[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
label =fish_df['Species'].to_numpy()
# print(fish_df)
# print(fish_df.columns)
# print(fish_df.info())
# print(fish_df.describe())

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 데이터 정규화,표준화작업
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)

# SGD : 분류 
# max_iter=100
sc = SGDClassifier(loss='log',random_state=42)

train_score=[]
test_score=[]
classes = np.unique(train_label) # 7개

# partial_fit : fit업데이트
# fit : 리셋후 다시 fit
# for _ in range(300):
#     # 분류해야 할것이 7개라는 것을 인지시켜줌.
#     sc.partial_fit(train_scaled,train_label,classes=classes)
#     train_score.append(sc.score(train_scaled,train_label))
#     test_score.append(sc.score(test_scaled,test_label))

# plt.plot(train_score)  
# plt.plot(test_score)  
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()

# 분류 - logistic손실함수
sc = SGDClassifier(loss='log',max_iter=100,tol=None,random_state=42)
sc.fit(train_scaled,train_label)
print("score train : ",sc.score(train_scaled,train_label))
print("score test : ",sc.score(test_scaled,test_label))
    
# 분류 - svm알고리즘손실함수
# sc = SGDClassifier(loss='hinge',max_iter=100,tol=None,random_state=42)
# sc.fit(train_scaled,train_label)
# print("score train : ",sc.score(train_scaled,train_label))
# print("score test : ",sc.score(test_scaled,test_label))



