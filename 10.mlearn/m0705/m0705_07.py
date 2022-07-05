from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit, softmax             # z점수 0-1사이의 값으로 변경
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
fish_df =pd.read_csv('10.mlearn/m0705/fish.csv')
data = fish_df[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
label = fish_df['Species'].to_numpy()

# 데이터 전처리
train_data,test_data,train_label,test_label=train_test_split(data,label,random_state=42)

# 정규화,표준화작업 - 손실값을 줄이는 형태구성.
ss=StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)

# -----------------------------------------------
# 확률적경사하강법
# epoch 300번 반복 train_score,test_score 그래프 출력
# sc = SGDClassifier(loss='log',random_state=42)
# train_score=[]
# test_score=[]
classes = np.unique(train_label)

# 300번반복 - partial_fit을 바로 사용시, class값을 전송
# for idx in range(300):
#     sc.partial_fit(train_scaled,train_label,classes=classes)
#     train_score.append(sc.score(train_scaled,train_label))
#     test_score.append(sc.score(test_scaled,test_label))

# plt.figure(figsize=(8,6))    
# plt.plot(range(300),train_score)
# plt.plot(range(300),test_score)
# plt.xlabel("epoch")
# plt.ylabel('accuracy')
# plt.show()    

# 확률적경사하강법사용    
sc = SGDClassifier(loss='log_loss',max_iter=100,tol=None,random_state=42)
# 훈련
sc.fit(train_scaled,train_label)
# sc.partial_fit(train_scaled,train_label)
# 정확도
score1 = sc.score(train_scaled,train_label)
score2 = sc.score(test_scaled,test_label)
print("train정확도 : ",score1)
print("test정확도 : ",score2)
