# 머신러닝-강화학습-분류-로지스틱회귀
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit, softmax             # z점수 0-1사이의 값으로 변경
# 시그모이드함수
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#class 7개 ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt'] 
#특성 5개 ['Weight','Length','Diagonal','Height','Width'] 
df_fish = pd.read_csv('10.mlearn/m0705/fish.csv')
data = df_fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
label = df_fish['Species'].to_numpy()  

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 정규화,표준화작업
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)
# -----------------------------------------------

# 로지스틱회귀 - 다중분류
# 알고리즘 선택
# 선형회귀 - 규제:릿지회귀-alpha제어
# 대문자, 기본값 C=1 C:낮으면 규제강함, C:높으면 규제약함

# 규제 score 추출 및 그래프그리기
# c_lists =[0.001,0.01,0.1,1,10,100] # 6개
# train_score=[]
# test_score=[]
# for c_list in c_lists:
#     lr = LogisticRegression(C=c_list)
#     # 알고리즘훈련
#     lr.fit(train_scaled,train_label)
#     train_score.append(lr.score(train_scaled,train_label))
#     test_score.append(lr.score(test_scaled,test_label))
    
# #그래프 그리기
# plt.plot(np.log10(c_lists),train_score)
# plt.plot(np.log10(c_lists),test_score)    
# plt.show()

# 알고리즘 선택
lr = LogisticRegression(C=10)
# 알고리즘 훈련
lr.fit(train_scaled,train_label)

# 예측
print("Bream Parkki Perch Pike Roach Smelt Whitefish")
proba = lr.predict(test_scaled[:5])
print(proba)

# 정확도
# score1 = lr.score(train_scaled,train_label)
# score2 = lr.score(test_scaled,test_label)
# print("정확도1 : ",score1)
# print("정확도2 : ",score2)

# z점수
decisions = lr.decision_function(test_scaled[:5])
# print(decisions)
# 로지스틱회귀-다중분류:소프트맥스함수, 시그모이드함수 이진분류
# axis=0 열방향, axis=1 행방향
print(np.round(softmax(decisions,axis=1),3))

print(0.+0.035+0.167+0.+0.066+0.731+0.)






