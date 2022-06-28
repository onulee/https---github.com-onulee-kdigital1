from logging import exception
from sklearn import svm,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 읽기---1
mr = pd.read_csv('./10.learn/agaricus-lepiota.data',header=None)

# 데이터 내부의 기호를 숫자로 취환---2
data = []
label = []
attr_list = []
for row_index,row in mr.iterrows():
    label.append(row[0])
    row_data = []
    for v in row[1:]:
        row_data.append(ord(v))   #ord('a')를 넣으면 정수 97을 반환
    data.append(row_data)  


# 분류변수, 연속변수 비교
# for row_index,row in mr.iterrows():   # p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
#     label.append(row[0])              # p
#     exdata = []
#     for col,v in enumerate(row[1:]):  # (0,x) x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
#         if row_index == 0:             # 0
#             attr = {"dic":{}, "cnt":0} # 
#             attr_list.append(attr)     # {"dic":{}, "cnt":0}
#         else:
#             attr = attr_list[col]
                
#         d = [0]*12
#         if v in attr['dic']:           # "dic":{}
#             idx = attr['dic'][v]
#         else:
#             idx = attr['cnt']          # attr[0]=0
#             attr['dic'][v] = idx
#             attr['cnt'] += 1
#             # raise exception('error')
#         d[idx] = 1
#         exdata += d
#     data.append(exdata)
    
# 학습데이터,테스트데이터 나누기---3
train_data,test_data,train_label,test_label = train_test_split(data,label)

# 데이터 학습시키기---4
clf = RandomForestClassifier()
clf.fit(train_data,train_label)

# 데이터 예측하기---5
predict = clf.predict(test_data)

# 결과 테스트---6
score = metrics.accuracy_score(test_label,predict)     
report = metrics.classification_report(test_label,predict)
print("정답률 : ",score) 
print("리포트 :\n ",report) 
# print(label[:5])
# print("-"*50)
print(data[:5]) 
       
     