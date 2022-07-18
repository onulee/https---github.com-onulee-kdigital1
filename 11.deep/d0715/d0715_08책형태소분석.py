from konlpy.tag import Okt
import pandas as pd

# book파일 가져오기
f = open('11.deep/d0715/book.txt',encoding='utf-8')
book = f.read()

#-------------------------------------------
# 형태소분석
okt = Okt()
# {'날카로운':1,'분석':10}
word_dic={}
lines = book.split("\n") # 1402
for line in lines:
    # 형태소 변환
    malist = okt.pos(line,norm=True,stem=True) 
    for taeso,pumsa in malist:
        if pumsa == 'Noun':
        # if pumsa in ['Noun','Josa','Verb']:
            # word_dic안에 형태소가 있는지 확인
            if not (taeso in word_dic):
                word_dic[taeso]=0
            word_dic[taeso] += 1   
#--------------------------------------------- 
# 1,2,1000,30,5,700
# 숫자역순정렬
keys = sorted(word_dic.items(),key=lambda x:x[1],reverse=True)

# 튜플 형태의 리스트로 출력
# 50개
for word,count in keys[:50]:
    print("{}:{} ".format(word,count),end=" ")
print()               
            
             





