import pandas as pd
import numpy as np
from sklearn import svm,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 글자 숫자로 변경함수
def nameChange(species):
    if species == 'setosa':
        return 0
    elif species == 'versicolor':
        return 1
    else:
        return 2

df = pd.read_csv('001.mach/iris.csv',index_col='caseno')
print(df.columns)             # 컬럼명 출력
print(df['Species'].unique()) # 중복된 row1개만 출력

data = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
# df['Species'] = df['Species'].apply(nameChange)  # 함수호출, 변경값:df 적용
label = df['Species']

print(data[0:120])

data_numpy = np.array(data) 
label_numpy = np.array(label) 
index = np.arange(49)
np.random.shuffle(index)

train_data = data_numpy[index[:35]]
test_data = data_numpy[index[35:]]
train_label = label_numpy[index[:35]]
test_label = label_numpy[index[35:]]

# # sklearn 라이브러리에서 랜덤으로 실습데이터,테스트데이터 분리 - default 0.2
# train_data,test_data,train_label,test_label = train_test_split(data,label)

# clf = KNeighborsClassifier()
# # clf = svm.SVC()
# clf.fit(train_data,train_label)
# result =clf.predict(test_data)
# print('결과값 : ',result)  # 0-setosa

# score = metrics.accuracy_score(result,test_label)
# score2 = clf.score(train_data,train_label)
# print('정답률 : ',score)
# print('정답률2 : ',score2)