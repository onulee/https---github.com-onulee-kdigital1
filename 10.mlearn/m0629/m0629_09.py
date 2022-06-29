data=[1.5,1.9,3.5,4.8,5.9]

# 함수호출1 : lambda 함수호출 lambda a : a 매개변수,int(a):실행문
map_data = list(map(lambda a: int(a),data))

# 함수호출2
def func(a):
    return int(a)
map_data = list(map(func,data))


print("data : ",data)
print("map데이터 결과값 : ",map_data)
    
