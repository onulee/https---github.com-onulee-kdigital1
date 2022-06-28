from sklearn import svm,metrics

# 데이터 처리
data = [[0,0], [1,0], [0,1], [1,1] ]
label = [ 0,1,1,0 ]
example = [[0,0],[1,0]]
example_label = [0,1]
# 데이터 학습시키기 
clf = svm.SVC()
clf.fit(data,label) # 데이터와 답은 list타입 

# 데이터 예측하기
results = clf.predict(example)
print(results)

# 정답률
score = metrics.accuracy_score(example_label, results)
print("정답률 : ",score)