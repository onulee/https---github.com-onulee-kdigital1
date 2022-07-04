from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 무게 105,길이 15.2, 대각선길이 19.5,높이 5.09, 두께 3.42 [105,15.2,19.5,5.09,3.42] 
# 물고기를 분류하시오. 
# KNN을 사용

# 데이터 불러오기 [159 rows x 6 columns]
df_fish = pd.read_csv('10.mlearn/m0704/fish.csv')
data = df_fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
label = df_fish['Species'].to_numpy()  #7가지['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt'] 

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 데이터 정규화,표준화작업
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)

# Bream,Smelt 물고기 남겨놓고 나머지는 삭제
# [[False,True,False,True.....]]
bream_smelt_index1 = (train_label == 'Bream') | (train_label == 'Smelt')
bream_smelt_index2 = (test_label == 'Bream') | (test_label == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_index1] # 33개
train_label = train_label[bream_smelt_index1] # 33개
test_bream_smelt = test_scaled[bream_smelt_index2]   # 16개
test_label = test_label[bream_smelt_index2] # 33개
print(train_bream_smelt.shape)
print(train_label.shape)
print(test_bream_smelt.shape)
print(test_label.shape)


# train_bream_smelt,test_bream_smelt,train_label,test_label


# 알고리즘 선택
# clf = KNeighborsClassifier()
lr = LogisticRegression()   
# 예측 - 선형회귀 lr.coef_ : 기울기, lr.intercept_ : y절편
#        예측값 : z = ax + b

# 실습훈련
lr.fit(train_bream_smelt,train_label)

# 예측
proba = lr.predict_proba(test_bream_smelt[:5])
print("전체예측결과 : ",np.round(proba,decimals=4))

# 정확도(정답률)
score1 = lr.score(train_bream_smelt,train_label)
score2 = lr.score(test_bream_smelt,test_label)

print("정확도1 : ",score1)
print("정확도2 : ",score2)





