import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings

# Data Source : https://www.blockchain.com/ko/charts/market-price?timespan=60days

file_path = '10.mlearn/m0708/market-price.csv'
bitcoin_df = pd.read_csv(file_path, names = ['day', 'price'])
print(bitcoin_df)

# 기본 정보를 출력합니다.
print(bitcoin_df.shape)
print(bitcoin_df.info())

bitcoin_df.tail()

# to_datetime으로 day 피처를 시계열 피처로 변환합니다. 
bitcoin_df['day'] = pd.to_datetime(bitcoin_df['day'])

# day 데이터프레임의 index로 설정합니다.
bitcoin_df.index = bitcoin_df['day']
bitcoin_df.set_index('day', inplace=True)
bitcoin_df.head()

bitcoin_df.describe()

# 일자별 비트코인 시세를 시각화합니다.
# bitcoin_df.plot()
# plt.show()

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA

# model = sm.tsa.arima.ARIMA(train_data, order=(1,1,2))
# result = model.fit()

# (AR=2, 차분=1, MA=2) 파라미터로 ARIMA 모델을 학습합니다.
model = ARIMA(bitcoin_df.price.values, order=(2,1,2),trend='n')

model_fit = model.fit()
print(model_fit.summary())

fig = model_fit.plot_predict() # 학습 데이터에 대한 예측 결과입니다. (첫번째 그래프)
residuals = pd.DataFrame(model_fit.resid) # 잔차의 변동을 시각화합니다. (두번째 그래프)
residuals.plot()
plt.show()