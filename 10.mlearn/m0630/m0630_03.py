from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob,os.path,re

# train,test폴더를 읽어오기 위해 함수생성
def makeData(url):
    # 디렉토리 내 모든 파일 읽어오기
    files = glob.glob(url)
    data = []  # 26개 a-z
    label = [] # 4개 en,fr,id,tl 
    for file_name in files: # 20개
        # 위치를 제외한 파일이름
        basename = os.path.basename(file_name)
        lang = basename.split('-')[0]
        
        # 파일읽어오기
        with open(file_name,'r',encoding='utf-8') as f:
            text = f.read() # 파일의 글을 추출
            text = text.lower()
            
            # 알파벳을 분류
            #  a,b,c,d, .......    z
            #  0 1 2 3             25
            # [1,1,0,0, .......    0]  # 26개
            code_a = ord('a')  # 97
            code_z = ord('z')  # 122
            count = [0]*26
            
            # text의 글을 1글자씩 분리
            for ch in text:
                code_current = ord(ch) # b 98-a=1, 99-a=2
                if code_a <= code_current <= code_z: # 97<=x<=122
                    count[code_current-code_a] += 1
            
            # 데이터 전처리
            total = sum(count) # 3만, 5천.... a 300/30000
            count = list(map(lambda n:n/total,count))
            data.append(count) 
            label.append(lang) 
            
    return data,label        
        

# ----------- 함수호출 ------------
# 1. 데이터 전처리
url1 = '10.mlearn/m0630/train/*.txt'
train_data,train_label = makeData(url1)

url2 = '10.mlearn/m0630/test/*.txt'
test_data,test_label = makeData(url2)


# 2. 알고리즘 선택
clf = svm.SVC()
# clf = KNeighborsClassifier()

# 3. 실습훈련
clf.fit(train_data,train_label)

# 4. 예측
result = clf.predict(test_data)
print("결과값 : ",result)

# 5. 정답률
score = clf.score(test_data,test_label)
print("정답률 : ",score)


 