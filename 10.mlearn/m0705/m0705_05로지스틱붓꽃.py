from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit, softmax             # z점수 0-1사이의 값으로 변경
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('10.mlearn/m0705/iris(150).csv',index_col='caseno')
data = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].to_numpy()
label = df['Species'].to_numpy()

# 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label,random_state=42)

# 정규화,표준화작업 -  5.1 3.8 1.8  0.5
ss = StandardScaler()
train_scaled = ss.fit_transform(train_data)
test_scaled = ss.fit_transform(test_data)
new_scaled = ss.fit_transform([[5.1,3.8,1.8,0.5]])

#---------------------------------------------------

# 알고리즘 선택
# clf = KNeighborsClassifier(n_neighbors=7)

# for문
# train_score = []
# test_score = []
# c_lists = [0.001,0.01,0.1,1,10,100]
# for c_list in c_lists:
#     lr = LogisticRegression(C=c_list)
#     lr.fit(train_scaled,train_label)
#     train_score.append(lr.score(train_scaled,train_label))
#     test_score.append(lr.score(test_scaled,test_label))
    
# plt.plot(np.log10(c_lists),train_score)    
# plt.plot(np.log10(c_lists),test_score)    
# plt.show()
    
# 알고리즘 선택    
lr = LogisticRegression(C=5,max_iter=1000)

# 훈련
lr.fit(train_scaled,train_label)

# 예측
result = lr.predict(new_scaled)
result2 = lr.predict_proba(new_scaled)
print("예측결과 : ",result)
print("예측결과proba : ",result2)

# 정확도
score1 = lr.score(train_scaled,train_label)
score2 = lr.score(test_scaled,test_label)
print("정확도 train : ",score1)
print("정확도 test : ",score2)

