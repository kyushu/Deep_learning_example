# -*- coding: utf-8 -*-

'''
USAEG:

    Shallow Net
    python train_network.py --network shallownet --model output/cifar10_shallownet.hdf5 --epochs 20

    LeNet
    python train_network.py --network lenet --model output/cifar10_lenet.hdf5 --epochs 20

    Karpathy Net
    python train_network.py --network karpathynet --model output/cifar10_karparthynet_without_dropout.hdf5 --epochs 100

    Karpathy Net: Enable dropout
    python train_network.py --network karpathynet --model output/cifar10_karparthynet_with_dropout.hdf5 --dropout 1 --epochs 100
'''


from __future__ import print_function
from mtImageSearch.cnn import ConvNetFactory
# SGD : Stochastic Gradient Descent
from keras.optimizers import SGD
# use keras.datasets to load cifar10 as Numpy array
from keras.datasets import cifar10
from keras.utils import np_utils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network",      required=True, help="name of network to build")
ap.add_argument("-m", "--model",        required=True, help="path to output model file")
ap.add_argument("-d", "--dropout",      type=int, default=-1,       help="whether or not dropout should be used")
ap.add_argument("-f", "--activation",   type=str, default="tanh",   help="activation function to use (LeNet only)")
ap.add_argument("-e", "--epochs",       type=int, default=20,       help="# of epochs")
ap.add_argument("-b", "--batch-size",   type=int, default=32,       help="size of mini-batches passed to network")
ap.add_argument("-v", "--verbose",      type=int, default=1,        help="verbosity level")
args = vars(ap.parse_args())


print("[INFO] loading training data ...")
# keras.datasets.cifar10.load_data() will return training part and test part (including data and labels)
((trainData, trainLabels), (testData, testLabels)) = cifar10.load_data()
# scale data into [0, 1] (data is pixel value)
trainData = trainData.astype("float") / 255.0
testData = testData.astype("float") / 255.0

# transform the training and testing labels into vectors in the range
# [0, numClasses] -- this generates a vector for each label, where the
# index of the label is set to `1` and all other entries to `0`; in the
# case of CIFAR-10, there are 10 class labels
# For instance:
# 如果 class 有 10 個，則 每個 Label 由單一個數字(代表第幾個 class ) 轉為一個 Vector
# vector index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# label = 7 => [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# label = 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)


kwargs = {"dropout": args["dropout"] > 0, "activation": args["activation"]}

# train model using SGD
print("[INFO] compiling model...")
# ConvNetFactory.build(network_name, numchannels, imgRows, imgCols, numClasses, keyword_arguments)
model = ConvNetFactory.build(args["network"], 3, 32, 32, 10, **kwargs)

# Before training your model, you need to configure the learning process by using "model.compile"
# model.compile takes three parameters : optimimzer, loss function, a list of metrics

# https://keras.io/optimizers/
# optimimzer 有很多種，目前先看 SGD = Stochastic Gradient Descent
# optimimzer : keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# lr        : float >= 0. Learning rate.
# momentum  : float >= 0. Parameter updates momentum.
# decay     : float >= 0. Learning rate decay over each update.
# nesterov  : boolean. Whether to apply Nesterov momentum.

# https://keras.io/objectives/
# loss function 有很多種
# 因為 cifar10 的 label 是用 one-hot 的方式，也就是 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 如果是 class 0 的 則 label = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 如果是 class 5 的 則 label = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# 所以 loss function(aka cost functiuon) 使用 categorical_crossentropy
# categorical_crossentropy: Also known as multiclass logloss.
#                           Note: using this objective requires that your labels are
#                           binary arrays of shape (nb_samples, nb_classes).

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


print("[INFO] starting training...")
print("trainData.shape:{}".format(trainData.shape[1:]))
print("batches_size:{}".format(args["batch_size"]))
print("epochs:{}".format(args["epochs"]))
model.fit(trainData, trainLabels, batch_size=args["batch_size"], nb_epoch=args["epochs"], verbose=args["verbose"])


# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=args["batch_size"], verbose=args["verbose"])
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# dump the network architecture and weights to file for we can reuse next time
print("[INFO] dumping architecture and weights to file...")
model.save(args["model"])


# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

# model = keras.Sequential() return a Squential model
# The Sequential model is a linear stack of layers

# You can use model.save(filepath) to save a Keras model into a single HDF5 file which will contain:
# the architecture of the model, allowing to re-create the model
# the weights of the model
# the training configuration (loss, optimizer)
# the state of the optimizer, allowing to resume training exactly where you left off.
#
# you can load model by model.load_model(filepath)
