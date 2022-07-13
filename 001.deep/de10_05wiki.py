import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from gensim.models import word2vec

model = word2vec.Word2Vec.load("001.deep/wiki2.model")


# most_similar 1개 출력,  positive : 긍정단어, negative : 부정단어 , topn : 5개출력
print(model.wv.most_similar(positive=['왕자','여성'], negative=['남성'], topn=5))
