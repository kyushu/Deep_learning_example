# -*- coding:utf-8 -*-



from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

seed = 7
np.random.seed(seed)
dataset = np.loadtxt("./dataset/pima-indians-diabetes.csv", delimiter=",")
x = dataset[:, 0:8]
y = dataset[:, 8]

# keras 提供 KerasClassifier() 將 neural network 包裝起來供 sklearn 使用
# 需先提供建立 nerual network 的 function
model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)

# sklearn 使用 keras 提供的 model 做 corss_validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())

