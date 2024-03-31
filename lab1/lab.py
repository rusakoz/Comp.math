import math
import os
import random
import re
import sys

# При 15 нейронах, 10к эпох, 0.1 альфе, 0.00001 бете
# Success! Wait for the results of the plagiarism check
# Test cases results (percentage):
# case 0: 100.0%
# case 1: 73.5%
# case 2: 92.6%
# case 3: 44.2%

#
# Complete the 'approximate_sigmoid' function below.
#
# The function is expected to return a DOUBLE.
# The function accepts following parameters:
#  1. DOUBLE_ARRAY x_axis
#  2. DOUBLE_ARRAY y_axis
#

# Возвращает результат ф-ии активации
def sigmoid(x):
    # Функция активации для X > 0: f(x) = 1 / (1 + exp^(-x))
    # Функция активации для X < 0: sigmoid(x) = 1 / (1 + exp(-x))
    #                                         = exp(x) / (exp(x) + 1)
    #                                         = 1 - 1 / (1 + exp(x))
    #                                         = 1 - sigmoid(-x)
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

# Возвращает результат производной ф-ии активации
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

# Генераторы весов и смещений:
# --- A ---
def generate_weights_1_2(quantity):
    xavier = math.sqrt(6) / math.sqrt(2 + 1)
    return [random.uniform(-xavier, xavier) for i in range(quantity)]

def generate_weights_2_3(quantity):
    xavier = math.sqrt(6) / math.sqrt(quantity + 1)
    return [random.uniform(-xavier, xavier) for i in range(quantity)]

def generate_bias():
    return random.uniform(-1.4, 1.4)
# --- A ---

# Модель нейронной сети
class NeuralNetwork():
    '''
    Нейронная сеть с:
        - Двумя входами(X и 1)
        - Скрытым слоем из N(задается значение) нейронов
        - Выходным слоем из 1-го нейрона
    Нейроны скрытого слоя имеют случайные веса и смещения:
        - Диапозон веса определяется формулой Ксавье([-x;x] x = sqrt(6) / sqrt(N + M), где N кол-во входов, M - выходов)
        - Диапозон смещения - [-1.4, 1.4]
    Нейрон выходного слоя имеет случайные веса:
        - Диапозон веса определяется формулой Ксавье([-x;x] x = sqrt(6) / sqrt(N + M), где N кол-во входов, M - выходов)
    '''
    def __init__(self, count_hidden_neurons):
        self.count_hidden_neurons = count_hidden_neurons
        self.hidden_neurons = [HiddenLayerNeuron(generate_weights_1_2(2), generate_bias()) for i in range(self.count_hidden_neurons)]
        self.output_neuron = OutputNeuron(generate_weights_2_3(self.count_hidden_neurons))
    
    def outputs_hidden_layer(self, inputs):
        return [self.hidden_neurons[i].output(inputs) for i in range(self.count_hidden_neurons)]

    def output_network(self, inputs):
        return self.output_neuron.output(inputs)
    
    def predict(self, inputs):
        return self.output_network([sigmoid(m) for m in(self.outputs_hidden_layer(inputs))])
    
    def max_err_predict(self, data_x, y_trues):
        deviation = 0
        for i in range(len(data_x)):
            inputs = [data_x[i], 1]
            predict = self.predict(inputs)

            error_predict = abs(predict - y_trues[i])

            if error_predict > deviation:
                deviation = error_predict
        return deviation
    
    def train(self, data_x, y_trues, epochs, alpha_weight, delta_bias):
        for i in range(epochs):
            for j in range(len(data_x)):
                inputs = [data_x[j], 1]

                # Получаем выход нейронной сети
                outputs_hidden_neuron = self.outputs_hidden_layer(inputs)
                sigmoid_hidden_neurons = [sigmoid(m) for m in(outputs_hidden_neuron)]
                predict = self.output_network(sigmoid_hidden_neurons)

                # Вычисляем ошибку 
                error_layer_3 = predict - y_trues[j]

                # Корректировка весов и смещений
                weight_delta_2_3 = 1 * error_layer_3
                for f in range(self.count_hidden_neurons):
                    # Корректировка весов выходного слоя
                    self.output_neuron.weights[f] -= alpha_weight * weight_delta_2_3 * sigmoid_hidden_neurons[f]
                    
                    # Корректировка весов скрытого слоя и смещений
                    error_layer_2 = error_layer_3 * 1 * self.output_neuron.weights[f]
                    gradient_layer_2 = diff_sigmoid(outputs_hidden_neuron[f])
                    weight_delta_1_2 = gradient_layer_2 * error_layer_2
                    self.hidden_neurons[f].weights[0] -= alpha_weight * inputs[0] * weight_delta_1_2
                    self.hidden_neurons[f].weights[1] -= alpha_weight * inputs[1] * weight_delta_1_2
                    self.hidden_neurons[f].bias -= delta_bias * gradient_layer_2

def approximate_sigmoid(x_axis, y_axis):
    # Кол-ва нейронов для простых ф-ий хвататет 15-25, но чем больше, тем точнее аппроксимация, но и обучать нужно дольше, 
    # кроме этого эпох нужно больше, в итоге время аппроксимации растет: epochs * count_neurons
    network = NeuralNetwork(count_hidden_neurons=15)
    # Больше изгибов ф-ии => больше delta_bias
    # alpha_weight обычно в диапазоне [0.1, 0.0001], нужно подбирать
    # Чем больше epochs, тем точнее аппроксимируется ф-ия, НО! Если правильно подобраны веса, смещения и их alpha_weight, delta_bias
    network.train(x_axis, y_axis, epochs=10000, alpha_weight=0.05, delta_bias=0.000001)
    return network.max_err_predict(x_axis, y_axis)

if __name__ == '__main__':
    axis_count = int(input().strip())

    x_axis = list(map(float, input().rstrip().split()))

    y_axis = list(map(float, input().rstrip().split()))

    result = approximate_sigmoid(x_axis, y_axis)

    print(str(result) + '\n')