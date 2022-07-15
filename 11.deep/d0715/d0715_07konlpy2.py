from konlpy.tag import Okt

okt = Okt()
# 형태소 모두 추출 : morphs
# print(okt.morphs('단독 일찰보다 복수 입찰일 경우'))
# 명사만 추출
# print(okt.nouns('유일하게 항공기 제작이 가능한 곳입니다.'))
# 글,품사 추출
# print(okt.pos('유일하게 항공기 제작이 가능한 곳입니다.'))
# 텍스트 어절 추출
# print(okt.phrases('날카로운 분석과 신뢰감 있는 진행으로'))
print(okt.pos('유일하게 항공기 제작이 가능한 곳입니다.'))
