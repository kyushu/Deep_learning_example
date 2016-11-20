# -*- coding:utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("./dataset/pima-indians-diabetes.csv", delimiter=",")
# row [0, 7] = x
X = dataset[:, 0:8]
# row [8] = y
Y = dataset[:,8]

# Create model: as a Sequence of layers
model = Sequential()
# Dense: full-connected layer
# input_dim: number of input variable and has to be set if it is the first layer
# init:      weight distribution, we use uniform distribution
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
# 根據 ~/.keras/keras.json 使用 theano or tensorflow 來設定(建立)每個 layer 跟 neuron
# 包含 loss(cost) function, gradient descent
# binary_crossentropy = logloss, 因為 Y 只有 0 跟 1
# 所以使用 binary_crosentropy
# categorical_crossentropy = multiclass logloss 也就是 Y 有 多個 class
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model (train model)
# 跑 150(nb_epoch) 次
# 每一次的 epoch 將 dataset 均分成每組有 10 (batch_size) 筆的 subset
# 每次執行一組 subset 做 (feedforwar + erro backpropagateion) = training process
# 所以一次完整的 epoch = 執行 (dataset / 10) 次 training
model.fit(X, Y, nb_epoch=150, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
