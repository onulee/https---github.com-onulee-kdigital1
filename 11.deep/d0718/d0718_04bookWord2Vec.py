from konlpy.tag import Okt
import pandas as pd
from gensim.models import Word2Vec

# book파일 가져오기
f = open('11.deep/d0718/book.txt',encoding='utf-8')
book = f.read()

# 네이버의 영화댓글
# 199992
# 1402
lines = book.split('\n')
# print(len(lines))


# # 형태소 분석
okt = Okt()
# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

token_data=[]
# 199992개를 가져와서 형태소 분석
for sent in lines:
    # 1줄을 가져와서 형태소 분석
    temp_x = okt.morphs(sent,stem=True) # 형태소 만들어져서 temp_x
    temp_x = [word for word in temp_x if not word in stopwords ]
    token_data.append(temp_x)
  
print(token_data[:10])  
  
    
# word2vec - 글과 글의 관계가 형성됨.   
model = Word2Vec(sentences=token_data,vector_size=100,window=5,min_count=5,workers=4,sg=0)
  
print("ok")

print(model.wv.most_similar(positive=['여자','결혼'],negative=['아빠']))    
