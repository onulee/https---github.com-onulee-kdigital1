import pandas as pd
# bert 자연어처리 모델
from sentence_transformers import SentenceTransformer
# 자연어vec 거리를 계산하는 함수
from sklearn.metrics.pairwise import cosine_similarity

# 구글에서 자연어처리 Bert 모델로 사용, 한국어 자연어처리
# SentenceTransformer에서 여러 모델중 1개
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 한국어 딥러닝 : 1. 형태소분석 2.불용어제거 3. 토큰화 4. 글을 숫자변경 5. 모델링
ex_text = ['안녕하세요.?', '한국어 문자 임베딩을 위한 버트 모델입니다.']
sample_embedding = model.encode(ex_text) 

print(sample_embedding)
