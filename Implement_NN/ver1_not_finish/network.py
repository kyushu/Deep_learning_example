# -*- coding:utf-8 -*-
import random
import numpy as np

# For example: net = Network([2, 3, 1])
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 產生隨機的 bias 跟 weight 作為初始值
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # sizes[:-1] 是指從 0 到 －1(不包含 － 1) 而 －1 == 最後一筆
        # sizes[:-1] 從 0 到最後一筆的前一筆
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        """ Return the output of the network if "a" is input. """
        for b, w in zip(self.biases, self.weights):
            # print("b.shape: {}".format(b.shape))
            # print("a.shape: {}".format(a.shape))
            a = sigmoid(np.dot(w, a)+b)
        return a

    # epochs : 一個完整 training data 餵給 neural network 的次數
    # training data : 一個大量完整的 training data
    # min_batch_size : 每次餵給 neural network training data 的數量
    # eta : learning rate
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent
        The "training_data is a list of tuples (x, y)" representing the training
            inputs and the desired outputs.
        The other non-optional parameters are self-explanatory.
        If "test_data" is provided then the network will be evaluated aginst the
            test data after each epch, and pratial progress printed out.
        This is useful for tracking progress but slows things down substantially.
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in xrange(epochs):
            # Stochastic gradient descent 第一步一定要先對 training data 做 randomize
            random.shuffle(training_data)

            # 根據 min_batch_size 將 training data 分成 min_batches 份
            mini_batches = [
                # 每 min_batch_size 為一組 training data
                training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)
            ]

            # mini_batch = (x, y);
            # x : training(test) data
            # y : label
            for mini_batch in mini_batches:
                # update basies, weights
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using
        backpropagation to as single mini batch.
        mini_batch : (x, y) = (data, label)
        eta        : learning rate
        """
        # allocate new array for update biases and wieghts
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # loop x:data and y:label to update biases and wieght
        for x, y in mini_batch:
            # backprop :
            #       1. feedforward x to the final layer, calculate the error at the final layer
            #       2. backpropagation error from "L-1" to "2" layer
            # delta_nabla_b =
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # 更新每個 layer 的 biases, weight
            # bias = b - (eta / m) * delta_nabla_b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # update wieghts
            # weight = weight - (eta / m) * delta_nabla_w
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (eta / len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 1. Feedforward
        # 第一筆輸入資料 len[x] = self.weights[0].shape[1] (column )
        activation = x
        # 存每個 layer 的 activation
        activations = [x]
        # zs: 存每個 layer 的 z vectors (neuron 的 input vector)
        # z = (W ⦁ X + B), W: weight, X: activation, B: intercept
        zs = []

        # 一開始已產生隨機數值的 bias, weight 作為初始值
        for b, w in zip(self.biases, self.weights):
            # 一次做 一整個 layer
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward
        # 先算出 final layer 的 error
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return(nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        result_data = [np.argmax(self.feedforward(x)) for (x, y) in test_data]
        desired_y = [np.argmax(y) for (x, y) in test_data]


        # print("len(result_data): {}".format(result_data))
        # print("len(desired_y): {}".format(desired_y))
        mttest = np.vstack( (result_data, desired_y) ).transpose()
        # print("result_data[3]:{}".format(result_data[3]))
        # print("desired_y[3]: {}".format(desired_y[3]))
        # print("mttest[3]: {}".format(mttest[3]))
        # f = open("desired_y", "w")
        # for (x, y) in mttest:
        #     f.write("x:{}, y:{}\n".format(x, y))
        # f.close()


        # print("self.feedforward(x): {}".format(test_results))
        # test_results = [ (np.argmax(self.feedforward(x)), int(y)) for (x, y) in test_data]
        # print("test_results: {}".format(test_results[0]))
        # mtresult = sum(int(x == y) for (x, y) in test_results)
        # print("x : {}".format(test_results[0]))
        return sum(int(x == y) for (x, y) in mttest)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)



### Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

def sigmoid_prime(z):
    """ Derivative of the sigmoid function. """
    return sigmoid(z)*(1-sigmoid(z))







