from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split # train,test데이터분리
from sklearn.preprocessing import StandardScaler     # 정규화,표준화작업
from scipy.special import expit, softmax             # z점수 0-1사이의 값으로 변경
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('10.mlearn/m0705/iris(150).csv',index_col='caseno')
data = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].to_numpy()
label = df['Species'].to_numpy()