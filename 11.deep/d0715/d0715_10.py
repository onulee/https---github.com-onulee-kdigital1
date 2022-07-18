from konlpy.tag import Okt
import pandas as pd
from bs4 import BeautifulSoup
import codecs

# 데이터 불러오기
fp = codecs.open("11.deep/d0715/BEXX0003.txt",'r',encoding='utf-16')
# soup = BeautifulSoup(fp,"lxml")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.get_text()

#-------------------------------------------
# 형태소분석
okt = Okt()
word_dic=[]
lines = text.split("\n") # 3440
for line in lines:
    # 형태소 변환
    malist = okt.pos(line,norm=True,stem=True)
    r=[] 
    for taeso,pumsa in malist:
        if len(taeso)>=2 :
            if pumsa == 'Noun':
                if not (taeso in word_dic):
                    word_dic[taeso]=0
                word_dic[taeso] += 1   
# --------------------------------------------- 
# 숫자역순정렬
keys = sorted(word_dic.items(),key=lambda x:x[1],reverse=True)

# 튜플 형태의 리스트로 출력
# 50개
for word,count in keys[:50]:
    print("{}:{} ".format(word,count),end=" ")
print() 