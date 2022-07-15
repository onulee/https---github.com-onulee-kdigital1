from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
csv = pd.read_csv('11.deep/d0715/bmi.csv')

# 원핫인코딩 : 팀명을 숫자로 변경 - 컬럼3개추가, 1개 삭제, 컬럼 5개
bmi_label = pd.get_dummies(csv['label'])
csv = csv.drop('label',axis=1)
csv = csv.join(bmi_label)

# 2. 데이터 전처리
# train 정규화,표준화 작업
csv['height'] = csv['height']/200
csv['weight'] = csv['weight']/100

# train,test데이터 분리
data = csv[['height','weight']]
label = csv[['fat','normal','thin']].to_numpy()

# train,test데이터
train_data,test_data,train_label,test_label = train_test_split(data,label)

# ---------------------------------
# 4. 딥러닝 선언
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(2,)))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

early_stop = keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)

# 5. 딥러닝 훈련
history = model.fit(train_data,train_label,\
    # batch_size=128,
    validation_data=(test_data,test_label),epochs=20,\
        callbacks=[early_stop])

print(model.summary())

# 그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.show()

# 5-2. (141,64) 예측(분류) 하시오.

val_labels = np.argmax(model.predict([[178/200,50/100]]),axis=-1)

print("result : ", val_labels)  # 0 1 2
# print("result : ", model.predict([[141/200,64/100]]))

# 6. 정확도 - score 
score = model.evaluate(test_data,test_label)
print(score)
