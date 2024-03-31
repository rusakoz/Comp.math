import math
import os
import random
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def ApproxFunc(x):
    return x**2

pcenter = 0
prange = 10
step = 0.1
period = np.arange(pcenter-prange, pcenter+prange, step)
# plt.plot(period, ApproxFunc(period))
# plt.show()

class PartyNN(object):

    def __init__(self, learning_rate=0.1, input_nodes=1, hidden_nodes=5, output_nodes=1):
        input_nodes+=1
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (hidden_nodes, input_nodes))
        self.weights_1_2 = np.random.normal(0.0, 1 ** -0.5, (output_nodes, hidden_nodes))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])
        # print("Первые веса \n", self.weights_0_1, "\nВторые веса ", self.weights_1_2)
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def predict(self, inputs):
        inputs = np.concatenate((inputs, [1]))
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = inputs_2
        return outputs_2

    def train(self, inputs, expected_predict):
        inputs = np.concatenate((inputs, [1]))
        # print("000: ", inputs)
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = inputs_2
        # print("111: ", outputs_2)
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = 1 # actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        # print("222: ", np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1))))
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate
        # print("333: ", self.weights_1_2)

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        # print("444: ", inputs)
        # print("555: ", inputs.reshape(len(inputs), 1))
        # print("666: ", np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T)
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate
        # print("777: ", self.weights_0_1)

lr = 0.0005
myNN = PartyNN(learning_rate=lr, input_nodes=1, hidden_nodes=5, output_nodes=1)

set_count = 100
rand_set = np.random.random(set_count)*2*prange-prange+pcenter

def Train():
    rand_set = np.random.random(set_count)*2*prange-prange+pcenter
    for x in range(0, set_count-1):
        myNN.train([rand_set[x]], ApproxFunc(rand_set[x]))
        # myNN.train([2], ApproxFunc(rand_set[x]))
print(np.random.normal(0.0, 2 ** -0.5, (5, 2)))
epochs = 1
plt.ion()
for e in range(epochs):
    for tr in range(100):
        Train()
    plt.clf()
    plt.plot(period, ApproxFunc(period), c="blue")
    plt.show()
    rand_set = np.random.random(set_count)*2*prange-prange+pcenter
    for x in range(0, set_count-1):
        plt.scatter(rand_set[x], myNN.predict([rand_set[x]]), c='red')
    plt.pause(1)



