import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ai데이터 : https://aihub.or.kr/
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
# 데이터 불러오기
df = pd.read_csv("C:/pydata/09.django/d0530_03_챗봇/sproject/fboard/wellness_dataset.csv")
df['embedding'] = pd.Series([[]] * len(df)) # dummy
# 모델인코딩 후, 유저컬럼 글자들이 embedding컬럼에 벡터화, list저장
df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))


# 개별적인 질문 함수호출
def return_answer(question):
    # 입력된 질문 벡터화
    embedding = model.encode(question)
    
    # 모든 질문과 입력된 질문의 유사도 저장
    df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    # print('similarity : ',df['similarity'])
    # # 가장 높은 유사도 글을 리턴
    answer = df.loc[df['similarity'].idxmax()]
    # answer = df.loc[3]
    print('구분 : ', answer['구분'])
    print('유사한 질문 : ', answer['유저'])
    print('챗봇 답변 : ', answer['챗봇'])
    print('유사도 : ', answer['similarity'])
    return answer

    