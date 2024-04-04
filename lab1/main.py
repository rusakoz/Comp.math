import math
import os
import random
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

#
# Complete the 'approximate_sigmoid' function below.
#
# The function is expected to return a DOUBLE.
# The function accepts following parameters:
#  1. DOUBLE_ARRAY x_axis
#  2. DOUBLE_ARRAY y_axis
#

def sigmoid(x):
    # Функция активации для X > 0: f(x) = 1 / (1 + exp^(-x))
    # Функция активации для X < 0: sigmoid(x) = 1 / (1 + exp(-x))
    #                                         = exp(x) / (exp(x) + 1)
    #                                         = 1 - 1 / (1 + exp(x))
    #                                         = 1 - sigmoid(-x)
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

def diff_sigmoid(x):
    # Производная функции активации f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

# Модель нейрона скрытого слоя
class HiddenLayerNeuron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    # Перемножаем веса(weights) со входами(inputs), суммируем, в конце добавляем смещение(bias) и возвращаем результат
    def output(self, inputs):
        return sum(self.weights[i] * inputs[i] for i in range(len(inputs))) + self.bias
    
# Модель нейрона выходного слоя
class OutputNeuron():
    def __init__(self, weights):
        self.weights = weights
    # Перемножаем веса(weights) со входами(inputs), суммируем и возвращаем результат
    def output(self, inputs):
        return sum(self.weights[i] * inputs[i] for i in range(len(inputs)))

def generate_weights_1_2(quantity):
    xavier = math.sqrt(6) / math.sqrt(2 + 1)
    return [random.uniform(-xavier, xavier) for i in range(quantity)]

def generate_weights_2_3(quantity):
    xavier = math.sqrt(6) / math.sqrt(quantity + 1)
    return [random.uniform(-xavier, xavier) for i in range(quantity)]

def generate_bias():
    return random.uniform(-1.4, 1.4)

class NeuralNetwork():
    '''
    Нейронная сеть с:
        - Двумя входами(X и 1)
        - Скрытым слоем из 3-х нейронов
        - Выходным слоем из 1-го нейрона
    Нейроны скрытого слоя имеют случайные веса и смещения:
        - Диапозон веса - [0.1, 2]
        - Диапозон смещения - [1, 2]
    Нейрон выходного слоя имеет случайные веса:
        - Диапозон веса - [0.1, 2]
    '''
    
    def __init__(self):
        self.count_hidden_neurons = 50
        self.hidden_neurons = [HiddenLayerNeuron(generate_weights_1_2(2), generate_bias()) for i in range(self.count_hidden_neurons)]
        self.output_neuron = OutputNeuron(generate_weights_2_3(self.count_hidden_neurons))
    
    def outputs_hidden_layer(self, inputs):
        return [self.hidden_neurons[i].output(inputs) for i in range(self.count_hidden_neurons)]

    def output_network(self, inputs):
        return self.output_neuron.output(inputs)
    
    def predict(self, inputs):
        return self.output_network([sigmoid(m) for m in(self.outputs_hidden_layer(inputs))])
    
    def train(self, data_x, y_trues):
        epochs = 6000
        alpha = 0.015
        delta = 0.005
        deviation = 0
        what_is = 0
        plt.ion()
        for i in range(epochs):
            for j in range(len(data_x)):
                inputs = [data_x[j], 1]

                # Получаем выход нейронной сети
                outputs_hidden_neuron = self.outputs_hidden_layer(inputs)
                sigmoid_hidden_neurons = [sigmoid(m) for m in(outputs_hidden_neuron)]
                predict = self.output_network(sigmoid_hidden_neurons)

                # Вычисляем ошибку 
                error_layer_3 = predict - y_trues[j]

                # Корректировка весов
                weight_delta_2_3 = 1 * error_layer_3
                for f in range(self.count_hidden_neurons):
                    # Корректировка весов выходного слоя
                    self.output_neuron.weights[f] -= alpha * weight_delta_2_3 * sigmoid_hidden_neurons[f]
                    
                    # Корректировка весов скрытого слоя
                    error_layer_2 = error_layer_3 * 1 * self.output_neuron.weights[f]
                    gradient_layer_2 = diff_sigmoid(outputs_hidden_neuron[f])
                    weight_delta_1_2 = gradient_layer_2 * error_layer_2
                    self.hidden_neurons[f].weights[0] -= alpha * inputs[0] * weight_delta_1_2
                    self.hidden_neurons[f].weights[1] -= alpha * inputs[1] * weight_delta_1_2
                    self.hidden_neurons[f].bias -= delta * gradient_layer_2

        for i in range(len(data_x)):
            inputs = [data_x[i], 1]
            predict = self.predict(inputs)

            error_predict = abs(predict - y_trues[i])

            if error_predict > deviation:
                deviation = error_predict
                what_is = data_x[i]

        print("MAX ERROR: ", deviation, what_is)


        def ApproxFunc(x):
            return x**2
        def sin(x):
            y = math.sin(x)
            return y
        def one_x(x):
            return 1 / x
        def trash(x):
            return 1 / x/2 + math.sin(x)/2 + math.cos(x)
        plt.clf()

        pcenter = 0
        prange = 10
        step = 0.1

        # Ломаная
        data_x = [0, 1, 2, 3, 4, 5, 6]
        y_trues = [0, 3, -3, 3, -3, 3, -3]
        plt.plot([0, 1, 2, 3, 4, 5, 6], y_trues)
        plt.show()
        data_x = np.random.random(100)*1*6

        # # Парабола
        # data_x = np.random.random(100)*2*10-10+0
        # y_trues = [data_x[i] ** 2 for i in range(len(data_x))]
        # period = np.arange(pcenter-prange, pcenter+prange, step)
        # plt.plot(period, ApproxFunc(period))
        # plt.show()

        # data_x = [-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,3,5,7,8,9,10,11,12,13,14,15,16]

        # # синус
        # f = np.vectorize(sin)
        # period = np.arange(-16, 16, step)
        # plt.plot(period, f(period))
        # plt.show()
        # data_x = np.random.random(1000)*2*16-16+0
        # y_trues = [sin(data_x[i]) for i in range(len(data_x))]

        # # 1/x
        # f = np.vectorize(one_x)
        # data_x = np.random.random(300)*1*9.8+0.1
        # y_trues = [1/data_x[i] for i in range(len(data_x))]
        # period = np.arange(0.1, 9.8, step)
        # plt.plot(period, f(period))
        # plt.show()

        # # trash
        # f = np.vectorize(trash)
        # data_x = np.random.random(100)*1*10+0.1
        # y_trues = [trash(data_x[i]) for i in range(len(data_x))]
        # period = np.arange(0.1, 10, step)
        # plt.plot(period, f(period))
        # plt.show()

        # plt.plot(data_x, y_trues, 'o', c='blue')
        # plt.show()

        if True:
            for m in range(len(data_x)):
                outputs_hidden_neuron = self.outputs_hidden_layer([data_x[m], 1])
                sigmoid_hidden_neurons = [sigmoid(m) for m in(outputs_hidden_neuron)]
                pred = self.output_network(sigmoid_hidden_neurons)
                plt.scatter(data_x[m], pred, c='red')
            plt.pause(900)
        outputs_hidden_neuron = self.outputs_hidden_layer([-10, 1])
        sigmoid_hidden_neurons = [sigmoid(m) for m in(outputs_hidden_neuron)]
        pred = self.output_network(sigmoid_hidden_neurons)
        print("ОТ КОНКРЕТНОГО ЗНАЧЕНИЯ ", pred)


def approximate_sigmoid(x_axis, y_axis):
    network = NeuralNetwork()
    # network.train([0,1,3,5,7,8,9,10,11,12,13,14,15,16], [0,0.841,0.1411,-0.9589,0.657,0.9893,0.41211,-0.544,-0.9999,-0.536,0.4201,0.9906,0.6502,-0.2879])
    # network.train([1, 1.1, 1.2, 1.3, 1.4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15.2, 15.4, 15.6, 15.8, 16], 
    #               [7, 6.9, 6.8, 6.7, 6.6, 5, 3, 5, 7, 9, 10, 11, 9, 8, 6, 5, 3, 5, 7, 7.2, 7.3, 7.5, 7.7, 9])
    # network.train([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 
    #               [13, 11, 10, 9, 8, 7, 6, 5, 4, 5, 6, 7, 8, 9, 10, 11, 15])
    # network.train([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
    #               [64, 49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64])
    # network.train([-8,-7.5, -7, -6.5, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8], 
    #               [64, 56.25, 49, 42.2, 36, 25, 16, 9, 4, 1, 0.25, 0, 0.25, 1, 4, 9, 16, 25, 36, 49, 64])

    # network.train([1, 2, 3], [10, 2, 10])

    # network.train([0, 1, 2, 3.1, 5, 6], [1, 2, 3.4, 4, 6, 3])
    # network.train([0.1, 0.3, 0.6, 1, 1.5, 2, 3.1, 5, 5.5, 6], [1, 13, 1.6, 2, 2.5, 3.4, 4, 6, 4, 30])

    # network.train([1, 1.1, 1.2, 1.3, 1.4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15.2, 15.4, 15.6, 15.8, 16,
    #                17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49], 
    #               [7, 6.9, 6.8, 6.7, 6.6, 5, 3, 5, 7, 9, 10, 11, 9, 8, 6, 5, 3, 5, 7, 7.2, 7.3, 7.5, 7.7, 9, 10, 11, 12, 13,
    #                9,8,7,6,5,4,7,8,9,10,11,12,13,14,13,12,11,10,9,8,7,6,5,4,3,2,3,4,5,6,7,8,9,10,11,12,13])

    # x_data = np.random.random(100)*2*10-10+0
    # y_data = [x_data[i] ** 2 for i in range(100)]

    # data_x = [1, 2.16336571, 6.71542695, -7.1268567, -3.72126273, 9.78995359,
    #         -7.6416726, 7.27217172, -6.25542576, 2.38561256, -9.47163364, 0.93529352,
    #         3.02920033, 6.36109399, 5.92167305, 5.44615816, -4.99810838, -8.18993883,
    #         4.49106521, 3.3149693, -4.80858852, -1.02612298, 9.0603214, 7.91642697,
    #         -5.563481, -3.34695029, 3.09651768, -7.55527048, -1.63663773, -7.01517414,
    #         5.98924419, -6.14352013, -3.60912236, -0.95432158, 9.133721, -3.03872207,
    #         -6.38712314, 6.08458524, -0.59484406, -8.42930016, 0.36337992, 2.66243181,
    #         8.65574282, -7.79463204, 2.17548514, 6.51467708, 0.04305243, 8.82460606,
    #         -3.3785624, 4.24376111, 4.30637679, 5.64859993, -5.16450603, 2.16994894,
    #         2.77329425, -3.78573791, 2.93362952, 3.98486508, -9.23429932, -3.51487684,
    #         9.55672713, -8.32933458, -8.6286195, -3.4396689, -0.58771289, -0.12829896,
    #         -9.77783166, -0.05551688, 6.7550489,  0.99413212, 1.46951828, -4.07828607,
    #         -6.32227549, -6.41148553, -0.07662755, 0.86086692, -6.95920028, 3.64038207,
    #         3.80500187, 4.93424515, 4.76284771, -2.28542858, 4.65333851, 6.09617539,
    #         -2.98413847, 4.31168867, 8.55228289, 6.34061672, -0.44677044, -9.68194157,
    #         1.66989391, 4.55245749, -7.22387749, 8.92143599, -4.15472793, -9.94482447,
    #         4.85769995, -8.34741248, 1.21770208, 3.63212024]

    # Ломаная
    data_x = [0, 1, 2, 3, 4, 5, 6]
    y_trues = [0, 3, -3, 3, -3, 3, -3]

    # # Парабола
    # # data_x = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # data_x = [-10, -2, 0, 2, 10]
    # # data_x = np.random.random(100)*2*10-10+0
    # y_trues = [data_x[i] ** 2 for i in range(len(data_x))]

    # # Синус
    # # data_x = [-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,3,5,7,8,9,10,11,12,13,14,15,16]
    # data_x = np.arange(-16.1, 16.1, 0.1)
    # # data_x = np.random.random(100)*2*16-16+0
    # y_trues = [math.sin(data_x[i]) for i in range(len(data_x))]

    # # 1/x
    # data_x = [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 2, 3.4, 5, 6.5, 7, 8.2, 9.8]
    # y_trues = [1/data_x[i] for i in range(len(data_x))]

    # # trash
    # # data_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 
    # #           2.1, 2.2, 2.3, 2.4, 2.5, 3.4, 4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 6.5, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6,
    # #             7.7, 7.8, 7.9, 8.2, 9.8]
    # data_x = np.arange(0.1, 10, 0.1)
    # y_trues = [1 / data_x[i] / 2 + math.sin(data_x[i])/2 + math.cos(data_x[i]) for i in range(len(data_x))]

    network.train(data_x, y_trues)
    # print(y_data)
    # network.train([1, 2, 3], [10, 2, 10])

    # r = [[3, 8], [2, 2]]
    

    # a = np.array([2 - 1])
    # inputs = [2, 2]
    # weights_1 = np.random.normal(0, 1, (10, 2))
    # inputs_1 = np.dot(weights_1, inputs)
    # print(weights_1)
    # print(inputs_1, inputs_1[0])
    # print(a * 2)

    # c = inputs_1.reshape(len(inputs_1), 1)
    # b = np.dot(c, [2]).T 

    # print(c, b)

    return 1


if __name__ == '__main__':
    # axis_count = int(input().strip())

    # x_axis = list(map(float, input().rstrip().split()))

    # y_axis = list(map(float, input().rstrip().split()))
    x_axis = 1
    y_axis = 2
    result = approximate_sigmoid(x_axis, y_axis)

    print(str(result) + '\n')