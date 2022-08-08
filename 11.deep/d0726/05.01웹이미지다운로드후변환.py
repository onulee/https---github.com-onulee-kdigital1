from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

# 이미지 데이터 다운로드
# http://www.vision.caltech.edu/datasets/
# https://drive.google.com/drive/folders/1cnQHqa8BkVx90-6-UojHnbMB0WhksSRc
# caltech-101 다운로드

# 1. 이미지 분류
caltech_dir = "11.deep/d0726/image/101_ObjectCategories"
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)
print(nb_classes) # 5

# 2. 이미지 크기 지정
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3
print(pixels) # 12288

# 3. 이미지 읽어오기
X = []
Y = []
for idx, cat in enumerate(categories): # 5개 "chair","camera","butterfly","elephant","flamingo"
    # 레이블 지정 원핫인코딩
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 현재위치 : "11.deep/d0726/image/101_ObjectCategories" 
    # cat : 5개 "chair","camera","butterfly","elephant","flamingo"
    image_dir = caltech_dir + "/" + cat
    # glob: 모든.jpg파일이름 리턴
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f) # --- (※6)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h)) #(64,64)
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

print(Y)





