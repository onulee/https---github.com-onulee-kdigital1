# m0629 > mushroom.csv
# e,p 식용버섯, 독버섯인지 구별하시오.
# 랜덤포레스트 알고리즘 사용할것

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split
import pandas as pd
import numpy as np

# 1.데이터 가져오기
df = pd.read_csv('10.mlearn/m0707/mushroom.csv')
data = df[df.columns.difference(['poisonous'])]  
# data = df[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']]
label = df['poisonous']
print(data.shape)

# 데이터 변환
new_data = []
for i in range(len(data)): #8124
    row_data=[] # 문자를 숫자로 변경한 데이터를 저장list
    # v 0-21
    for v in range(len(data.iloc[i])):  # 컬럼 22개
        row_data.append(ord(data.iloc[i,v]))
    new_data.append(row_data)    

data = new_data

# 2. 데이터 전처리
train_data,test_data,train_label,test_label = train_test_split(data,label)


# 알고리즘 선택
# cpu core개수를 몇개를 사용할지 정함. -1 core를 사용함.
# 랜덤포레스트 : 결정트리사용 디폴트 10개사용, n_estimators=100
rf = RandomForestClassifier(n_jobs=-1,random_state=42)

# 교차검증훈련
# return_train_score = True : train_score가 출력
scores = cross_validate(rf,train_data,train_label,return_train_score=True,n_jobs=-1)

print("train_score 평균 : ",np.mean(scores['train_score']))
print("test_score 평균 : ",np.mean(scores['test_score']))

# 4. 실습훈련
rf.fit(train_data,train_label)

# 5. 예측
result = rf.predict(test_data)
print('결과값 : ',result) 

# 6. 정답률
score1 = rf.score(train_data,train_label)
score2 = rf.score(test_data,test_label)
print("정답률1 : ",score1)
print("정답률2 : ",score2)


        
        
        
    
