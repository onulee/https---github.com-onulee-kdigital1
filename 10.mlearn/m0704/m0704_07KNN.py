from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
new_scaled = ss.fit_transform([[105,15.2,19.5,5.09,3.42]])


# 알고리즘 선택
clf = KNeighborsClassifier()

# 실습훈련
clf.fit(train_scaled,train_label)

# 예측
# result = clf.predict([[105,15.2,19.5,5.09,3.42]])
result = clf.predict(test_scaled[:5])
print("예측결과 :",result)

proba = clf.predict_proba(test_scaled[:5])
print("전체예측결과 : ",np.round(proba,decimals=4))

# 정확도(정답률)
score1 = clf.score(train_scaled,train_label)
score2 = clf.score(test_scaled,test_label)

print("정확도1 : ",score1)
print("정확도2 : ",score2)





