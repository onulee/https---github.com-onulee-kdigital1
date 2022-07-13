import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd

# 소설책 읽어오기
f= open('001.deep/book.txt',encoding='utf-8')
book = f.read()
# print(book) # 전체글 출력

okt = Okt()
word_dic = {}
lines = book.split("\r\n")
# 1줄씩 가져와서 for반복문
for line in lines:
    # 형태소 분석
    malist = okt.pos(line, norm=True , stem=True)
    # 형태소, 품사분리
    for taeso, pumsa in malist :
        # 명사일 경우만 추가
        if pumsa == "Noun":
            # word_dic안에 형태소가 없으면 list추가후 1증가
            if not (taeso in word_dic):
                word_dic[taeso] = 0
            word_dic[taeso] += 1

# 숫자역순정렬        
keys = sorted(word_dic.items(),key=lambda x:x[1],reverse=True)        
# 50개 데이터 정렬
for word,count in keys[:50]:
    print("{0}({1})".format(word,count),end="")
print()