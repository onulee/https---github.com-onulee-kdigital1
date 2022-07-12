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
for line in lines:
    malist = okt.pos(line, norm=True , stem=True)
    for taeso, pumsa in malist : 
        if not (taeso in word_dic):
            word_dic[taeso] = 0
        word_dic[taeso] += 1

# 숫자역순정렬        
keys = sorted(word_dic.items(),key=lambda x:x[1],reverse=True)        
# 50개 데이터 정렬
for word,count in keys[:50]:
    print("{0}({1})".format(word,count),end="")
print()