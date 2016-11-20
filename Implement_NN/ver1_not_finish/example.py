import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()




net = network.Network([784, 30, 10])
# print("test_data.shape: {}; test_data[0]: {}".format(len(test_data), test_data[0]))
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


