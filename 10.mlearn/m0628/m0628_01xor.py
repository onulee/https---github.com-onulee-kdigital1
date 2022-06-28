from cProfile import label
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# 1. 데이터 전처리
train_data = [[0,0],[0,1],[1,0],[1,1]]
train_label = [0,1,1,0]

example = [[0,0],[1,0]]  # 0,1

# 1-2. 데이터 처리
# 훈련데이터와 테스트데이터를 분리를 해야 함.
test_data = [[0,1],[1,1]]
test_label = [1,0]


# 2. 알고리즘 선택
clf = svm.SVC() #머신러닝 SVC알고리즘 선택
# clf = KNeighborsClassifier() # KNN알고리즘 선택

# 3. 학습시키기
clf.fit(train_data,train_label)

# 4. 데이터 예측
result = clf.predict(example)
print("결과값 : ",result)

# 5. 정답률
score = clf.score(test_data,test_label)
print("정답률 : ",score)
