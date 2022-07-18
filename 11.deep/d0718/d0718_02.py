# 형태소 분석

from konlpy.tag import Okt

# 게시판, 댓글 글을 적으면, 너무맛있어요.완전짱!정말음식이끝내줘요.
# 100000단어 -> 데이터 전처리 단계

okt = Okt()
text = '한글 자연어 처리는 재밌다. 이제부터 열심히 해야지 ㅎㅎㅎ'
# 텍스트 단위로 형태소 추출
print(okt.morphs(text))

# 명사만 추출
print(okt.nouns(text))

# 어절단위로 추출
print(okt.phrases(text))

# 품사도 함께 추출 (튜플형태)
print(okt.pos(text))
# 품사와 함께 추출
print(okt.pos(text,join=True))