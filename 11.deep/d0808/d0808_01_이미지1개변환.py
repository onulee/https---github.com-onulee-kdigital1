# 데이터 파일 1개를 웹에서 가져와서 변형
# Pillow 이미지 변형
from PIL import Image
# 디렉토리 찾아서 파일을 읽어올때
import os 
import numpy as np

# 파일 불러오기
with open('11.deep/d0808/b1.jpg','rb') as file:
    img = Image.open(file)
    # 흑백파일 변경
    img = img.convert('L')
    img = img.resize((28,28))
    # 255 숫자로 변환
    data = np.array(img)
    
# print(data.shape)  # (28,28)
# 딥러닝에 사용될수 있는 데이터 전처리
data = data/255
 
