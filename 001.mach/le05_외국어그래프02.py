from logging import exception
from sklearn import svm,metrics
import glob, os.path, re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus']=False
import pandas as pd

def makeData(url):
    # 폴더안 파일명 확인
    files = glob.glob(url)  # print(files)
    # files = glob.glob("./train/*.txt")  # print(files)
    data = []
    label = []
    for file_name in files:
        basename = os.path.basename(file_name)
        # print(basename) # basename은 파일명만, file_name 경로까지 나옴.
        lang = basename.split("-")[0]
        # print(basename, lang)
        with open(file_name,'r',encoding='utf-8') as f:
            text = f.read() # 파일안 글 추출.
            text = text.lower() # 소문자 변환
            # print(text)
            # 알파벳 출현 빈도 구하기
            code_a = ord('a')
            code_z = ord('z')
            count = [0 for n in range(26)]    # [0,0,0...] 26개
            for ch in text:
                # print(ch)
                code_current = ord(ch)
                if code_a <= code_current <= code_z:
                    count[code_current - code_a] += 1  # 'b'98 - 'a'97 = 1
            total = sum(count)
            count = list(map(lambda n:n / total,count))
            data.append(count)
            label.append(lang) 
            
    return data,label 

url1 = "10.learn/train/*.txt"        
train_data,train_label = makeData(url1) 

print(train_data)

# 그래프 준비하기
graph_dict = {}
for i in range(0,len(train_label)):
    label = train_label[i]
    data = train_data[i]
    if not (label in graph_dict):
        graph_dict[label] = data
 
asclist = [[chr(n) for n in range(97,97+26)]]
print(asclist) 

df = pd.DataFrame(graph_dict,index=asclist)

# 그래프 그리기
plt.style.use('ggplot')
df.plot(kind='bar',subplots=True,ylim=(0,0.15))
plt.savefig('lang-plot.png')
plt.show()

  
            
        