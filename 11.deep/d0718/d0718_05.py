from konlpy.tag import Okt
import pandas as pd
from gensim.models import Word2Vec

# 토지의 책 불러오기
# 소설책 읽어오기
f= open('11.deep/d0718/book.txt',encoding='utf-8')
book = f.read()
# 형태소분석 각각의 단어 몇번씩 나오는지 출력
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
# keys = sorted(word_dic.items(),key=lambda x:x[1],reverse=True)        
# # 50개 데이터 정렬, 빈도가 3이하의 숫자 제외
# for word,count in keys[:50]:
#     if count > 3:
#         print("{0}({1})".format(word,count),end="")
# print()    
     
# Word2Vec 벡터화

# 김서방 : 연관글을 찾아보시오.