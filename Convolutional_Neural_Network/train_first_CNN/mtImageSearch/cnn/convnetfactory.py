# -*- coding: utf-8 -*-

from keras.layers.convolutional import Convolution2D # CONV Layer
from keras.layers.convolutional import MaxPooling2D  # POOL Layer
from keras.layers.core import Activation             # Activation Function like RELU
from keras.layers.core import Flatten                # Full-Connected Layer
from keras.layers.core import Dense                  # Full-Connected Layer
from keras.layers.core import Dropout
from keras.models import Sequential                  # concatenate layers


class ConvNetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build(name, *pargs, **kwargs):
        # Predefine network function / name pair
        mappings = {
            "shallownet": ConvNetFactory.ShallowNet,
            "lenet": ConvNetFactory.LeNet,
            "karpathynet": ConvNetFactory.KarpathyNet,
            # "minivggnet": ConvNetFactory.MiniVGGNet
        }
        # get network from "name" which is input
        builder = mappings.get(name, None)

        if builder is None:
            return None

        # input arguments to configure network function
        return builder(*pargs, **kwargs)

    #
    # ShallowNet : INPUT => CONV => RELU => FC
    #
    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        # initialize the model (concatenate all layers)
        # 利用 keras 建立 CNN 就是使用 keras.models.Sequential (feedforward)
        model = Sequential()

        #
        # 1. Add First Layer : "CONV -> RELU" layer
        #
        # k = 32
        # receptive field = 3x3
        model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(imgRows, imgCols, numChannels)))
        # model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(numChannels, imgRows, imgCols)))

        model.add(Activation("relu"))

        #
        # 2. Add Second Layer: FC Layer
        #
        # First,  we need to flatten out multi-dimensional network into a 1D list
        #           by adding a Flatten() call to the network
        model.add(Flatten())
        # second, We then define a Dense()  fully-connected layer using the supplied number of classes
        #           (in the case of CIFAR-10, we have a total number of ten possible classes).
        model.add(Dense(numClasses))

        #
        # 3. Apply softmax activation (i.e., Logistic Regression)
        #
        # the final is softmax classifier
        model.add(Activation("softmax"))

        return model

    #
    # LeNet : INPUT => CONV => TANH => POOL => CONV => TANH => POOL => FC => TANH => FC
    #
    @staticmethod
    def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):

        model = Sequential()

        # First set of CONV => ACTIVATION => POOL Layers
        model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(imgRows, imgCols, numChannels)))
        model.add(Activation(activation))
        # Non-overlapping max-pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation(activation))
        # Non-overlapping max-pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # The First FC => ACTIVATION Layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # The Second FC Layer
        model.add(Dense(numClasses))

        # The Final layer soft-max classifier
        model.add(Activation("softmax"))

        return model

    #
    # KarpathtyNet
    #
    # 這裡我只是以實作這個架構的人 (Andrej Karpathy) 來命名
    # KarpathyNet : INPUT => (CONV => RELU => POOL => (DROPOUT?)) * 3 => SOFTMAX
    @staticmethod
    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        model = Sequential()

        # 1. First set of CONV => RELU => POOL layers
        model.add(Convolution2D(16, 5, 5, border_mode="same", input_shape=(imgRows, imgCols, numChannels)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # 2. Second set of CONV => RELU => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout:
            model.add(Dropout(0.25))

        # 3. Third set of CONV => RELU => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout:
            model.add(Dropout(0.5))

        # 4. The softmax classifier
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model


    #
    # MiniVGGNet : Input => CONV => RELU => CONV => RELU => POOL => FC => RELU => FC => SOFTMAX
    #
    @staticmethod
    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):

        model = Sequential()

        # first set of CONV => RELU => CONV => RELU => POOL layers
        model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(imgRows, imgCols, numChannels)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        # strides = (1, 1)
        # because pool_size = (2, 2) but stride = (1, 1) so it is overlapping pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))


        # Second set of CONV => RELU => CONV => RELU => POOL layers
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(63, 3, 3))
        model.add(Activation("relu"))
        # because pool_size = (2, 2) but stride = (1, 1) so it is overlapping pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # Third set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.5))

        # Final the soft-max classifier
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model











# def function_name(arg1, arg2, *pargs, **kwargs):
# arg1, arg2 是一般參數
# *pargs : 是 positional arguments, 也就是在 arg1, arg2 之後並且沒有參數名的都是 positional arguments
#          *pargs 是 tuple
# **kwargs: 是 keyword arguments, 也就是在 arg1, arg2 之後並且有參數名的都是 keyword arguments
#          **kwargs 是 dictionary
