# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("./dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 使用 sklearn.model_selection.train_test_split 將資料分成 training and test dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 設定 validation＿data
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=150, batch_size=10)

