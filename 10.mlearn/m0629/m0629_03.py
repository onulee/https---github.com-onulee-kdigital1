from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 1. 데이터 가져오기 - mushroom.csv 1개/22개
df = pd.read_csv('10.mlearn/m0629/mushroom.csv')
# poisonous,cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat
data = df[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]
label = df['poisonous']

a_data=[]
for i in range(len(data)):    # 8124
    row_data=[]
    for v in range(len(data.iloc[i])): # 22 
        row_data.append(ord(data.iloc[i,v]))
    a_data.append(row_data)  
    
data = a_data    

    
# data = []
# label = []
# # iterrows -> 2개 리턴 (index,list형태의 데이터)
# for index,rows in df.iterrows():
#     # 0, [p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u]
#     # 1, [e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g]
#     label.append(rows[0])
#     row_data=[]
#     for v in rows[1:]:
#         row_data.append(ord(v)) # 아스키코드 변환후 저장
#     data.append(row_data) 

# 2. 데이터 전처리
# train_data,test_data,train_label,test_label
train_data,test_data,train_label,test_label=\
    train_test_split(data,label,stratify=label)

# 3. 알고리즘 선택
clf = KNeighborsClassifier()

# 4. 실습훈련
clf.fit(train_data,train_label)

# 5. 예측
result = clf.predict(test_data)
print("결과값 : ",result)

# 6. 정답률
score1 = clf.score(train_data,train_label)
score2 = clf.score(test_data,test_label)
print("정답률1 : ",score1)
print("정답률2 : ",score2)