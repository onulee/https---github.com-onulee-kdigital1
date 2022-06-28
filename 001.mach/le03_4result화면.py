from sklearn import model_selection, svm, metrics
import pandas as pd
train_csv = pd.read_csv("./mnist/train.csv",header=None)
tk_csv = pd.read_csv("./mnist/t10k.csv",header=None)


def test(l):
    output=[]
    for i in l:
        output.append(float(i)/256)
    return output

# 모든 row, 1열 부터 모두 가져옴.
train_csv_data = list(map(test,train_csv.iloc[:,1:].values))
# 모든 row, 1열 부터 모두 가져옴.
tk_csv_data = list(map(test,tk_csv.iloc[:,1:].values))
# print(tk_csv_data)
train_csv_label = train_csv[0].values 
tk_csv_label = tk_csv[0].values

clf = svm.SVC()
# fit에 들어갈 데이터는 0~1사이의 값이어야 함.
clf.fit(train_csv_data,train_csv_label)
predict = clf.predict(tk_csv_data)
score = metrics.accuracy_score(tk_csv_label,predict)
print('정답률 : ',score)