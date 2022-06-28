from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# pandas데이터 불러오기
df = pd.read_csv('10.mlearn/m0628/iris(150).csv')

sLength = float(input("SepalLength 데이터를 입력하시오.(4.3 - 7.9)>>"))
sWidth = float(input("SepalWidth 데이터를 입력하시오.(2.0 - 4.4)>>"))
pLength = float(input("PetalLength 데이터를 입력하시오.(1.0 - 6.9)>>"))
pWidth = float(input("PetalWidth 데이터를 입력하시오.(0.1 - 2.5)>>"))

# # 1. 데이터 전처리 ( 120개-train데이터,30개-test데이터 )
# # train데이터, test데이터를 특정한 데이터에 치우지지 않도록 분리
# # 데이터를 섞어서 분리를 하도록 하시오.
# # 150개 random 섞어서 120개-train_data, 30개-test_data 분리를 하시오.
# df = df[['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']]
# df_numpy = df.to_numpy()
# np.random.shuffle(df_numpy)
# train_data = df_numpy[:120,:4]
# train_label = df_numpy[:120,4:5]
# print(train_label)
# print(train_label.shape)
# test_data = df_numpy[120:,:4]
# test_label = df_numpy[120:,4:5]
# print(test_label)
# print(test_label.shape)

# ----------------------------------------------
data = df[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
label = df['Species']

# 데이터전처리 : 80%,20%로 랜덤으로 데이터를 분리해줌.
# test_size : test데이터를 30% 랜덤으로 분리
# random_state=42 데이터를 랜덤으로 섞을때 규등하게 섞음
# stratify : 데이터가 한쪽으로 쏠려서 분배되지 않도록 해줌.
train_data,test_data,train_label,test_label = \
    train_test_split(data,label,test_size=0.3,stratify=label,random_state=42)


# # numpy 배열변경
# data_numpy = np.array(data) # 150개
# label_numpy = np.array(label)

# # 0-149까지의 랜덤 index만들기
# index = np.arange(150) # 150개 배열생성
# np.random.shuffle(index)
# print(index)
# # index를 활용해서 120개,30개 데이터 분리
# train_data = data_numpy[index[:120]]
# train_label = label_numpy[index[:120]]
# test_data = data_numpy[index[120:]]
# test_label = label_numpy[index[120:]]


# 2. 알고리즘 선택
# clf = svm.SVC()
clf = KNeighborsClassifier()

# 3. 데이터 학습훈련-학습훈련데이터
clf.fit(train_data,train_label)

# 4. 데이터 예측
result = clf.predict([[sLength,sWidth,pLength,pWidth]])
print("결과값 : ",result)

# 5. 정답률 - test데이터 들어감.
score = clf.score(test_data,test_label)
print("정답률 : ", score)