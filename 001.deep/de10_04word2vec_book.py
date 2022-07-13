import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from gensim.models import word2vec

# 소설책 읽어오기
f= open('001.deep/book.txt',encoding='utf-8')
book = f.read()
# 텍스트를 한 줄씩 처리하기 --- (※2)
okt = Okt() 
results = [] 
lines = book.split("\r\n")
for line in lines:
    # 형태소 분석하기 --- (※3)
    # 단어의 기본형 사용
    malist = okt.pos(line, norm=True, stem=True)
    r = []
    for word,pumsa in malist:
        # 어미/조사/구두점 등은 대상에서 제외 - 조사,어미,기호가 아니면
        if not pumsa in ["Josa", "Eomi", "Punctuation"]:
            r.append(word)
    rl = (" ".join(r)).strip()
    results.append(rl) 
    print(rl)
# 파일로 출력하기  --- (※4)
gubun_file = 'book.gubun'
with open(gubun_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))
# Word2Vec 모델 만들기 --- (※5)
# 1. 문장을 넣어서 분리
data = word2vec.LineSentence(gubun_file)
# 2. 분리한 문장을 넣어서 word2Vec으로 변환
model = word2vec.Word2Vec(data, 
    vector_size=200, window=10, hs=1, min_count=2, sg=1)
model.save("book.model")
print("ok")


# most_similar 1개 출력,  positive : 긍정단어, negative : 부정단어 , topn : 5개출력
print(model.wv.most_similar(positive=['여자'], negative=['남자'], topn=5))

# 10개 출력
print(model.wv.most_similar_cosmul(positive=['여자']))