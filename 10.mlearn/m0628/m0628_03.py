from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 4개의 입력을 만들어서
# 입력된 데이터를 가지고, 해당품종을 분류하는 프로그램을
# 구축하시오.


df = pd.read_csv('10.mlearn/m0628/iris(150).csv')
print(df.describe())

sepalLength = input("SepalLength 데이터를 입력하시오.(4.3 - 7.9)>>")
sepalWidth = input("SepalWidth 데이터를 입력하시오.(2.0 - 4.4)>>")
petalLength = input("PetalLength 데이터를 입력하시오.(1.0 - 6.9)>>")
petalWidth = input("PetalWidth 데이터를 입력하시오.(0.1 - 2.5)>>")
print("1번째 데이터 : ",sepalLength)
print("1번째 데이터 : ",sepalWidth)
print("1번째 데이터 : ",petalLength)
print("1번째 데이터 : ",petalWidth)






