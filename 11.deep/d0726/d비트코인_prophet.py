from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split # train,test
from sklearn.preprocessing import PolynomialFeatures, StandardScaler     # 정규화,표준화작업
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from fbprophet import Prophet


# 1. 데이터 불러오기
file_path = 'deeplearning//market-price.csv'
bitcoin_df = pd.read_csv(file_path, names = ['day', 'price'])

print(bitcoin_df.shape)
print(bitcoin_df.info())


# 2. 데이터 확인
# 일반 머신러닝은 train_data, test_data를 분리해서 러닝후 예측
# 시계열 예측은 특정 시점을 가지고 데이터를 분리
# 이전까지의 10일 train_data, next 5일을 test_data로 분리
# tail 향후 5일이 됨.
bitcoin_df.tail()

# to_datetime으로 day 컬럼타입을 date로 변경. 
bitcoin_df['day'] = pd.to_datetime(bitcoin_df['day'])
 
# day컬럼을 df index로 설정. - x좌표로 활용하기 좋음
bitcoin_df.index = bitcoin_df['day']
bitcoin_df.set_index('day', inplace=True)
bitcoin_df.head()
bitcoin_df.describe()

# 일자별 비트코인 시세를 시각화.
# bitcoin_df.plot()
# plt.show()

#----------------------------------------------------------------
# 파이썬 라이브러리를 활용한 시세 예측 - [ARIMA 모델 활용하기]