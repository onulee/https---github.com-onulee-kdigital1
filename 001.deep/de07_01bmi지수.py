from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

csv = pd.read_csv('11.deep/bmi.csv')
csv['weight'] /= 100
csv['height'] /= 200
X = csv[['weight', 'height']].values
# X = csv[['weight', 'height']].as_matrix()

bclass = {'thin': [1, 0, 0], 'normal': [0, 1, 0], 'fat': [0, 0, 1]}
y = np.empty((20000, 3))
for i, v in enumerate(csv['label']):
    y[i] = bclass[v]

# train데이터, test데이터 분리
X_train, y_train = X[1:15001], y[1:15001]
X_test, y_test = X[15001:20001], y[15001:20001]

# 모델생성
model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# 클래스 3개
model.add(Dense(3))
model.add(Activation('softmax'))

# 모델설정 - categorical_corssentropy 다중분류
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# 훈련데이터 진행
hist = model.fit(
    X_train, y_train,
    batch_size=100,
    epochs=6,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1
)

# 검증데이터 진행
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1]) 