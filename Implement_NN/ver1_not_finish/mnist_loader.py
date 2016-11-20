# -*- coding:utf-8 -*-


# Third-party libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import numpy as np

def load_data():
    mnist = fetch_mldata('MNIST original')
    (train_data, intermediateData, trainLabel, intermediateLabel) = train_test_split(mnist.data, mnist.target, test_size=0.5, random_state=42)
    (test_data, validation_data, testLabel, validationLabel) = train_test_split(intermediateData, intermediateLabel, test_size=0.2, random_state=84)

    # return  "Training dataset", "Validation dataset", "Test dataset"
    return ((train_data, trainLabel), (validation_data, validationLabel), (test_data, testLabel))

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()
    # training dataset
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # convert label into Numpy-array for [0, 1, ..., 9]
    # each element means each class
    training_labels = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_labels)

    # validation dataset
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_labels = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_labels)

    # test dataset
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_labels = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_labels)

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
