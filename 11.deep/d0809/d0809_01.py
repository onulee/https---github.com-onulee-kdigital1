import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# 한글설정-그래프한글표시
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['font.family'] = 'Apple Gothic' # apple사용시
matplotlib.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings("ignore")

# 파일 불러오기
rating_data = pd.io.parsers.read_csv('11.deep/d0809/movie/ratings.dat', names=['user_id', 'movie_id', 'rating', 'time'], delimiter='::') 
movie_data = pd.io.parsers.read_csv('11.deep/d0809/movie/movies.dat', names=['movie_id', 'title', 'genre'], delimiter='::') 
user_data = pd.io.parsers.read_csv('11.deep/d0809/movie/users.dat', names=['user_id', 'gender', 'age', 'occupation', 'zipcode'], delimiter='::')

# 평점정보 : user_id,movie_id,rating
print(rating_data.head(2))

# 영화정보 : movie_id,title,genre
print(movie_data.head(2))

# 회원정보 : user_id,gender,age
print(user_data.head(2))

# null 데이터가 있는지 확인
print(rating_data.isnull().sum())
print(movie_data.isnull().sum())
print(user_data.isnull().sum())

# 년도별 영화개수 확인
# movie_data['title'][10]
print(movie_data['title'][0][-5:-1])

# 총 영화 개수 - 3883개
print(len(movie_data['movie_id'].unique()))

# 총 회원수 - 6040명
print(len(user_data['user_id'].unique()))

# movie_data year컬럼 추가
# apply() : 함수적용
movie_data['year'] = movie_data['title'].apply(lambda x:x[-5:-1])
print(movie_data[:2])

# 년도별 영화개수 출력
# 년도별 영화개수 - 높은순으로 정렬되어서 출력
print(movie_data['year'].value_counts().head())

# 연대별 출력 ( 1910,1920,1930....)
movie_data['year_term'] = movie_data['title'].apply(lambda x : x[-5:-2]+'0')
print(movie_data.head(2))

# 년대별 영화개수 출력
movie_year_term = movie_data['year_term'].value_counts().sort_index()

# pip install seaborn
# sns.barplot(movie_year_term.index,movie_year_term.values,alpha=0.8)
# plt.title('년대별 영화수')
# plt.ylabel('개수')
# plt.xlabel('년도')
# plt.show()

# 장르별 검색 - movie_data : genre
print(movie_data.head(2))

# Action이 포함되어 있는 영화 개수
# movie_data : genre-Action : 503개
print(len(movie_data[movie_data['genre'].str.contains('Action')]))


# Action 만 있는 개수
data = movie_data['genre'].apply(lambda x : x=='Action')
print(len(movie_data[data]))

print(movie_data['genre'].value_counts()[:10])

# 장르별 영화 개수
print(movie_data.columns)

print(movie_data['genre'][:5])

test1 = movie_data['genre'][0].split('|')
print(test1) # list타입


# Drama:843, Comedy:521
unique_genre_dict={}
for index,row in movie_data.iterrows():
    genre_combination = row['genre'] # Animation|Children's|Comedy
    # list타입으로 받음.
    parsed_genre = genre_combination.split('|')
    
    # 장르별 분류
    for genre in parsed_genre: #['Animation', "Children's", 'Comedy']
        if genre in unique_genre_dict: #dic타입
            unique_genre_dict[genre] += 1
        else:
            unique_genre_dict[genre] = 1    

print(unique_genre_dict)   

# 장르별 그래프
# plt.rcParams['figure.figsize']=[20,16]
# sns.barplot(list(unique_genre_dict.keys()),list(unique_genre_dict.values()),alpha=0.8)
# plt.ylabel('장르개수')
# plt.xlabel('장르')
# plt.show() 



# [퀴즈]

# * 회원의 성별 분류를 하시오.
# * 회원의 연령대 분류를 하시오. 
print(user_data.columns)  


# 회원의 성별 분류
user_gender = user_data['gender'].value_counts()
print(user_data['gender'].value_counts())

# 성별그래프
# plt.rcParams['figure.figsize']=[4,4]
# sns.barplot(user_gender.index,user_gender.values)
# plt.show()

# 연령대별 분류
print(user_data['age'].value_counts())

# 10대,20대.... 분류
def age_class(age):
    if age==1:
        return 'etc'
    else:
        return str(age)[0]+'0' # 15-> 1+'0' -> 10

# 10대,20대,30대...
user_data['ages'] = user_data['age'].apply(lambda x : age_class(x))
user_ages = user_data['ages'].value_counts().sort_index() 
# 연령대별 인원수 
print(user_data['ages'].value_counts().sort_index())


# 연령대별 그래프
# plt.rcParams['figure.figsize']=[4,4]
# sns.barplot(user_ages.index,user_ages.values)
# plt.show()

# 평점 데이터 정보
# 영화별 평점이 주어진 개수
movie_rate_count = rating_data.groupby('movie_id')['rating'].count().values
# 영화평점 그래프
# plt.rcParams['figure.figsize']=[8,8]
# fig = plt.hist(movie_rate_count, bins=200)
# plt.ylabel('개수')
# plt.xlabel('영화평점 개수')
# plt.show()

print('총 영화 개수 : ',len(movie_data['movie_id'].unique()))
print('평점을 받은 개수가 100개이하인 수 : ',len(movie_rate_count[movie_rate_count<100]))


# 영화별 평점을 받은 수, 평균 평점
movie_grouped_rating_info = rating_data.groupby('movie_id')['rating'].agg(['count','mean'])
movie_grouped_rating_info.columns=['rated_count','rating_mean']
print(movie_grouped_rating_info.head())


print(movie_data[:1])

# 평균 평점 그래프
# plt.rcParams['figure.figsize']=[8,8]
# fig = plt.hist(movie_rate_count, bins=200)
# plt.ylabel('개수')
# plt.xlabel('영화평점 개수')
# plt.show()

# 주피터노트북에서 그래프출력 방법
# movie_grouped_rating_info['rating_mean'].hist(bins=150,grid=False)
# vscode에서 그래프출력 방법
# plt.hist(movie_grouped_rating_info['rating_mean'], bins=150)
# plt.show()

# 평점 개수가 100개이상이면서, 평점이 높은 10개를 출력하시오.
# 평점 개수가 100개 이상이면서 평점이 높은 순으로 10개 출력
# 영화평점개수 : movie_grouped_rating_info['rating_count']
# 영화제목 : movie_data['title']
# 해당 컬럼이 다른곳에 있어서, 컬럼을 합치기를 해야 함.
# movie_grouped_rating_info 있는 것만 생성
merged_data = movie_grouped_rating_info.merge(movie_data,on=['movie_id'],how='left')
print(merged_data[merged_data['rated_count']>100][['movie_id','rated_count','rating_mean','title']].nlargest(10,'rating_mean'))


# [퀴즈]

# * 유저별로 평가한 영화 개수 : 
# 1번,2번,3번,.. 유저가 몇개 평점을 했는지 개수를 출력
user_grouped_rating_info = rating_data.groupby('user_id')['rating'].agg(['count','mean','std'])
user_grouped_rating_info.columns=['rated_count','rating_mean','rating_std']
print(user_grouped_rating_info.head())


# rating_mean 그래프 출력
# 주피터노트북에서 출력방법
# user_grouped_rating_info['rating_mean'].hist(bins=150,grid=False)
# vscode에서 출력방법
# plt.hist(user_grouped_rating_info['rating_mean'],bins=150)
# plt.show()

print(rating_data.head())

rating_table = rating_data[['user_id','movie_id','rating']].set_index(['user_id','movie_id']).unstack()
print(rating_table)

from surprise import SVD,Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# 과학적 표기법 억제 # 1.500e-10 .500e+00  -> 0. 1.5 
np.set_printoptions(suppress=True)

# 실습 예제 : 테스트
rating_dict = { 'item_id':[1,2,3,1,2,3,2],
                'user_id':['a','a','a','b','b','b','c'],
                'rating':[2,4,4,2,5,4,5]
}
df = pd.DataFrame(rating_dict)
print(df)

df_matrix_table = df[['user_id','item_id','rating']].set_index(['user_id','item_id']).unstack().fillna(0)
print(df_matrix_table)

# SVD를 이용한 NaN 데이터 채우기
# SVD에 넣을수 있는 데이터셋을 생성
# 평점의 범위 지정 : 1-5점까지 범위지정
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df=df[['user_id','item_id','rating']],reader=reader)
# 행렬완성 데이터셋으로 변경
train_data = data.build_full_trainset()

# SVD모델 훈련
model = SVD(
    n_factors=8,
    lr_all=0.005, # 모든 파라미터 학습비율
    reg_all=0.02, # 모든 파라미터 정규화 정도
    n_epochs=10
)
model.fit(train_data)

# SVD를 활용해서 값 
# build_anti_testset : 행렬에서 채워지지 않은 위치를 가져옴
test_data = train_data.build_anti_testset()
# 채워지지 않은 위치에 값을 예측
predictions = model.test(test_data)

# 예측한 값 출력
for _, iid, _, predicted_rating, _ in predictions:
    print(('item_id :',iid,"|",'예측값 :',predicted_rating))


# 전체 데이터 예측값 출력
test_data = train_data.build_testset()
predictions = model.test(test_data)

for _, iid, real_rating,predicted_rating, _ in predictions:
   print('item_id :',iid,'|','실제평점 :',real_rating,'|','예측평점 :',predicted_rating) 


# 실제 데이터 적용
# * rating_data[['user_id','movie_id','rating']] 사용

# SVD 데이터셋 생성
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(rating_data[['user_id','movie_id','rating']],reader)
train_data = data.build_full_trainset()

# SVD모델 훈련
train_start = time.time()
model = SVD(
    n_factors=8,
    lr_all=0.005, # 모든 파라미터 학습비율
    reg_all=0.02, # 모든 파라미터 정규화 정도
    n_epochs=10
)
model.fit(train_data)
train_end = time.time()
print('모델 훈련시간 : %.2f 초'%(train_end - train_start))

# user_id 4번 회원의 영화평점이 없는 영화의 예측 평점을 구하시오.
# 4번 회원의 영화평점한 영화 개수
target_user_id = 4
target_user_data = rating_data[rating_data['user_id']==target_user_id]
print(target_user_data)

# 영화마다 평점을 출력
# user_id 4번이 평점을 매긴 영화만 저장
target_user_movie_rating_dict={}
for index,row in target_user_data.iterrows():
    movie_id = row['movie_id']
    target_user_movie_rating_dict[movie_id]=row['rating']
    
print(target_user_movie_rating_dict)


# user_id 4번인 회원이 평점을 하지 않은 영화
test_data = []
# 모든영화데이터에서 평점을 매기지 않은 영화를 찾기
for index,row in movie_data.iterrows():
    movie_id = row['movie_id']
    rating = 0
    # 영화평점이 있는 데이터
    if movie_id in target_user_movie_rating_dict:
        continue # 스킵
    # target_user_id:4, 영화, 0:평점이 매겨지지 않은 값
    test_data.append((target_user_id,movie_id,rating))
    
    
print(test_data[:5]) 

# 평점이 없는 영화 평점 예측
target_user_predictions = model.test(test_data)

# predictions:예측점수포함,user_history:평점이 된 영화
def get_user_predicted_ratings(predictions,user_id,user_history):
    # 예측 평점
    target_user_movie_predict_dict ={}
    # uid:user_id,mid:movie_id,rating:실제평점, 예측평점
    for uid,mid,rating,predicted_rating,_ in predictions:
        # 4번 회원것만 찾음
        if user_id == uid:
            # 평점이 없는 것만 찾음
            if mid not in user_history:
                target_user_movie_predict_dict[mid] = predicted_rating
    return target_user_movie_predict_dict 

# 함수호출 : 평점이 없는 영화데이터의 예측평점을 가져옴.
target_user_movie_predict_dict = get_user_predicted_ratings(
    # 평점이 없는 영화데이터 저장
    predictions=target_user_predictions,
    # 4번
    user_id = target_user_id,
    # 평점이 매개져 있는 영화데이터만 저장
    user_history = target_user_movie_rating_dict
)           


import operator

target_user_top10_predicted = sorted(
    target_user_movie_predict_dict.items(),
    # key movie_id,rating,  평점기준
    key=operator.itemgetter(1),
    # 역순정렬 10개
    reverse=True)[:10]  

print(target_user_top10_predicted) 

# 영화제목과 예측평점을 출력
# movie_id,title 2개를 dic타입으로 저장
movie_dict={}
for index,row in movie_data.iterrows():
    movie_id = row['movie_id']
    movie_title = row['title']
    movie_dict[movie_id] = movie_title
    
    
# 예측 평점 상위10개 출력
# 영화제목,예측평점 출력
# target_user_top10_predicted : movie_id,rating
print('영화번호','영화제목','예측평점')
for predicted in target_user_top10_predicted:
    movie_id = predicted[0]
    predicted_rating = predicted[1]
    print(movie_id,movie_dict[movie_id],':',predicted_rating)    
    

# user_id 4번이 평점한 영화 리스트
print(target_user_movie_rating_dict)

# 실제 평점이 높은 10개 영화데이터 출력
target_user_top10_real = sorted(target_user_movie_rating_dict.items(),
                                key=operator.itemgetter(1),reverse=True)[:10]
print(target_user_top10_real)

# 실제평점 상위10개 출력
# 영화번호,영화제목,실제평점 출력
# target_user_top10_real : movie_id,rating
print('영화번호','영화제목','예측평점')
for real in target_user_top10_real:
    movie_id = real[0]
    real_rating = real[1]
    print(movie_id,movie_dict[movie_id],':',real_rating)
    
# 예측 모델 정확도 측정
# SVD 데이터셋 생성
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(rating_data[['user_id','movie_id','rating']],reader)
# train_data = data.build_full_trainset()
train_data,test_data = train_test_split(data,test_size=0.2)

# SVD모델 훈련
train_start = time.time()
model = SVD(
    n_factors=8,
    lr_all=0.005, # 모든 파라미터 학습비율
    reg_all=0.02, # 모든 파라미터 정규화 정도
    n_epochs=10
)
model.fit(train_data)
train_end = time.time()
print('모델 훈련시간 : %.2f 초'%(train_end - train_start))

# 모델 정확도 측정
predictions = model.test(test_data)

# RMSE : 제곱근 오차
print('[ SVD모델 데이터셋 RMSE ]')
accuracy.rmse(predictions)


# 실제평점과 예측평점의 비교
# 4번회원의 실제 평점이 주어진 영화데이터 찾기
test_data=[]
# 모든 영화데이터 가져오기
for index,row in movie_data.iterrows():
   # 실제평점이 있는것만 가져옴.
   movie_id = row['movie_id']
   # 4번이 쓴 평점 데이터 모음과 비교 : target_user_movie_rating_dict
   if movie_id in target_user_movie_rating_dict:
       rating = target_user_movie_rating_dict[movie_id]
       test_data.append((target_user_id,movie_id,rating))  
       
print(test_data)          


# 4번이 평점을 매긴 영화데이터를 가지고 예측평점 훈련
# 실제평점,예측평점 비교할 목적
target_user_predictions = model.test(test_data)

# predictions:예측점수포함,user_history:평점이 된 영화
def get_user_predicted_ratings(predictions,user_id,user_history):
    # 예측 평점
    target_user_movie_predict_dict ={}
    # uid:user_id,mid:movie_id,rating:실제평점, 예측평점
    for uid,mid,rating,predicted_rating,_ in predictions:
        # 4번 회원것만 찾음
        if user_id == uid:
            # 평점이 없는 것만 찾음
            if mid in user_history:
                target_user_movie_predict_dict[mid] = predicted_rating
    return target_user_movie_predict_dict 

# 함수호출 : 평점이 없는 영화데이터의 예측평점을 가져옴.
target_user_movie_predict_dict = get_user_predicted_ratings(
    # 평점이 없는 영화데이터 저장
    predictions=target_user_predictions,
    # 4번
    user_id = target_user_id,
    # 평점이 매개져 있는 영화데이터만 저장
    user_history = target_user_movie_rating_dict
)  


print(target_user_movie_predict_dict)


# 예측평점, 실제평점 영화타이틀을 출력
origin_rating_list=[]
predicted_rating_list=[]
movie_title_list=[]
idx=0

# 실제평점의 예측평점 데이터
for movie_id,predicted_rating in target_user_movie_predict_dict.items():
    idx = idx + 1
    # 예측평점 소수점 2자리
    predicted_rating = round(predicted_rating,2)
    # 실제평점 데이터를 찾아 저장
    origin_rating = target_user_movie_rating_dict[movie_id]
    # 영화번호를 가지고 영화제목 찾아 저장
    movie_title = movie_dict[movie_id]
    print('movie',str(idx),':',movie_title,'-',origin_rating,'/',predicted_rating)
    
    # list저장
    origin_rating_list.append(origin_rating)
    predicted_rating_list.append(predicted_rating)
    movie_title_list.append(str(idx))
    
print(movie_title_list)

# 그래프 시각화
origin = origin_rating_list
predicted = predicted_rating_list

plt.rcParams['figure.figsize']=(10,6)
# 실제평점 21개
index = np.arange(len(movie_title_list))
bar_width=0.2

# 실제평점과 예측평점 bar그래프 출력
rects1 = plt.bar(index,origin,bar_width,color='orange',label='Origin')
rects2 = plt.bar(index+bar_width,predicted,bar_width,color='green',label='Predicted')
plt.xticks(index,movie_title_list)
plt.legend()
plt.show()    
    
    
