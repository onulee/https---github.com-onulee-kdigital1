from wordcloud import WordCloud
import matplotlib.pyplot as plt
import wordcloud

text="파이썬 파이썬 파이썬 워드클라우드 워드클라우드 라이브러리 좋아 좋아 예시 워드클라우드 워드클라우드 데이터분석 데이터 분석\
    파이썬 파이썬 파이썬 파이썬 딥러닝 딥러닝 딥러닝 머신러닝"
    
wordcloud = WordCloud('11.deep/d0718/NANUMGOTHIC.TTF').generate(text) 
# interpolation='bilinear' : 글자부드럽게
plt.imshow(wordcloud,interpolation='bilinear') 
plt.axis('off')
plt.show()
  