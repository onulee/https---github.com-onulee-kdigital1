from konlpy.tag import Okt
okt = Okt()

# 형태소 모두 추출
print(okt.morphs('단독입찰보다 복수입찰의 경우'))
# 명사만 추출
print(okt.nouns('유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))
# 텍스트 어절 추출
print(okt.phrases('날카로운 분석과 신뢰감 있는 진행으로'))
# 품사별 추출
print(okt.pos('이것도 되나욬ㅋㅋ'))
# norm : ㅋㅋ까지 추출
print(okt.pos('이것도 되나욬ㅋㅋ', norm=True))
# stem : 되나요-> 되다 어근 추출
print(okt.pos('이것도 되나욬ㅋㅋ', norm=True, stem=True))