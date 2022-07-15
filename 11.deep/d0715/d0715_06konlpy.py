from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt

# konlpy 선언
okt = Okt()
# pos : 형태소분석, norm : 그래욬ㅋㅋ -> 그래요 , stem 잘나가는 -> 잘나가다
malist = okt.pos('아버지 가방에 들어가신다',norm=True,stem=True)
print(malist)