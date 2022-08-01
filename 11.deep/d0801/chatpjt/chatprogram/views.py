import json
from django.http import HttpResponse
from django.shortcuts import render

#-------------------------------------------
import pandas as pd
# 챗봇 : 한국어 자연어처리 모델
from sentence_transformers import SentenceTransformer
# 기존데이터와 입력된 데이터의 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity

# 한국어 자연어처리 모델을 가져옴. : 구글 BERT
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
# 데이터 불러오기 - 심리상담 데이터
df = pd.read_csv('C:/pydata/11.deep/d0801/wellness_dataset_original.csv')
# 필요없는 컬럼 삭제 - Unnamed: 3 컬럼 삭제
df = df.drop(columns=['Unnamed: 3'])
df = df[~df['챗봇'].isna()]

# 1034 1차원 행렬로 변경
df['embedding'] = pd.Series([[]]*len(df)) # 1034
# df['유저']
df['embedding'] = df['유저'].map(lambda x: list(model.encode(x)))

#-------------------------------------------

def chat(request):
    return render(request,'chat.html')

# chat창에서 글을 입력받아, 챗봇데이터를 찾아 리턴해서 돌려줌.
# json데이터로 데이터 가져오기
from django.views.decorators.csrf import csrf_exempt

# json에서 POST넘기려면 꼭 넣어줘야 함.
@csrf_exempt
def chat_service(request):
    
    if request.method == 'POST':
        # json 넘어온 데이터
        input1 = request.POST['input1']
        print('json에서 넘어온 데이터 : ',input1)
    
    # 리턴타입 : dic타입
    output = dict()
    
    
    return HttpResponse(json.dumps(output),status=200)
    
