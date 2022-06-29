from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1.데이터 가져오기
df = pd.read_csv('10.mlearn/m0629/mushroom.csv')
data = df[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]
label = df['poisonous']


# 2. 데이터 전처리
# train_data,test_data,train_label,test_label
# train_data,test_data : 2차원 배열, 데이터 숫자여야 함.

new_data = []
for i in range(len(data)): #8124
    row_data=[] # 문자를 숫자로 변경한 데이터를 저장list
    # v 0-21
    for v in range(len(data.iloc[i])):  # 컬럼 22개
        row_data.append(ord(data.iloc[i,v]))
    new_data.append(row_data)    

data = new_data

train_data,test_data,train_label,test_label = train_test_split(data,label)


# 3. 알고리즘 선택
# clf = KNeighborsClassifier()
clf = svm.SVC()

# 4. 실습훈련
clf.fit(train_data,train_label)

# 5. 예측
result = clf.predict(test_data)
print('결과값 : ',result) 

# 6. 정답률
score1 = clf.score(train_data,train_label)
score2 = clf.score(test_data,test_label)
print("정답률1 : ",score1)
print("정답률2 : ",score2)


        
        
        
    
