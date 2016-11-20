
# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("./dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 這裡使用 keras 內建的 validation_split 將 training data (X) 的 0.33％ 分成 validation dataset
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)

