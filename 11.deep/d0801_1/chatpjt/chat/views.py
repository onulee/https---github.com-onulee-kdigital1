import json
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# ---------------------------------------
# 챗봇 알고리즘 start

import pandas as pd
# bert 자연어처리 모델
from sentence_transformers import SentenceTransformer
# 자연어vec 거리를 계산하는 함수
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
df = pd.read_csv('C:/pydata/11.deep/d0801_1/wellness_dataset_original.csv')
df = df.drop(columns=['Unnamed: 3'])
df = df[~df['챗봇'].isna()]

# df : 1034개 1차원 행렬
df['embedding'] = pd.Series([[]]*len(df))
# 유저컬럼의 데이터를 embedding해서 저장
df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

def def_answer(input_text):
    # 예시 질문
    em_text = model.encode(input_text)
    # em_text와 embedding의 거리를 측정해서 확률로 환산
    df['distance'] = df['embedding'].map(lambda x:cosine_similarity([em_text],[x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    
    return answer

# 챗봇 알고리즘 end
# ---------------------------------------

def chat(request):
    return render(request,'chat.html')

# json post방식으로 받을때 입력
@csrf_exempt
def chat_service(request):
    if request.method =='POST':
        # 입력값
        input = request.POST['input1']
        print('views전달값 : ',input)
        
        # -----------------------------------------
        # model 적용
        answer = def_answer(input)
        # -----------------------------------------
        
        # dic타입
        output=dict()
        print('유저질문 : ',answer['유저'])
        output['response'] = answer['챗봇']
        output['distance'] = answer['distance']
        return HttpResponse(json.dumps(output),status=200)
