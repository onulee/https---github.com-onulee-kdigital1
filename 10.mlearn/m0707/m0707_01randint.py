from scipy.stats import uniform,randint
rgen = randint(2,100)  # 정수
print(rgen.rvs(5))

# ugen = uniform(0,1)  # 실수
# print(ugen.rvs(10))