# -*- coding:utf-8 -*-

from random import seed
from random import random

def initialize_network(n_inputs, n_hidden, n_outputs):
    network= list()
    # n_inputs     = a1 + ... + an
    # n_inputs + 1 = b + a1 + ... + an
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)

    output_layer = [{'weights': [random() for i in range(n_hidden +1)]} for i in range(n_outputs)]
    network.append(output_layer)

    return network

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)

