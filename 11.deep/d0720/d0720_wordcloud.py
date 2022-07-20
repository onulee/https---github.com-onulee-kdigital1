from konlpy.tag import Okt
import pandas as pd
from gensim.models import Word2Vec
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# book파일 가져오기
f = open('11.deep/d0720/book.txt',encoding='utf-8')
book = f.read()

# print(book)

#-------------------------------------------
# 형태소분석
okt = Okt()
malist = okt.nouns(book) 

data = ' '.join(malist)
   
wordcloud = WordCloud('11.deep/d0718/NANUMGOTHIC.TTF').generate(data) 
# interpolation='bilinear' : 글자부드럽게
plt.imshow(wordcloud,interpolation='bilinear') 
plt.axis('off')
plt.show()