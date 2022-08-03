from konlpy.tag import Okt
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


#-------------------------------------------
# 형태소분석
okt = Okt()
# {'날카로운':1,'분석':10}
text = "coffee phone phone phone phone phone phone phone phone phone cat dog dog"
malist = okt.nouns(text) 

data = ' '.join(malist)
print(data)

wordcloud = WordCloud('11.deep/d0718/NANUMGOTHIC.TTF').generate(data) 
# interpolation='bilinear' : 글자부드럽게
plt.imshow(wordcloud,interpolation='bilinear') 
plt.axis('off')
plt.show()
  

             
            
             





