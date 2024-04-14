import math
import os
import random
import re
import sys

class Result:
    
    def first_function(x: float, y: float):
        return math.sin(x)


    def second_function(x: float, y: float):
        return (x * y)/2


    def third_function(x: float, y: float):
        return y - (2 * x)/y


    def fourth_function(x: float, y: float):
        return x + y

    
    def default_function(x:float, y: float):
        return 0.0

    # How to use this function:
    # func = Result.get_function(4)
    # func(0.01)
    def get_function(n: int):
        if n == 1:
            return Result.first_function
        elif n == 2:
            return Result.second_function
        elif n == 3:
            return Result.third_function
        elif n == 4:
            return Result.fourth_function
        else:
            return Result.default_function

    #
    # Complete the 'solveByRungeKutta' function below.
    #
    # The function is expected to return a DOUBLE.
    # The function accepts following parameters:
    #  1. INTEGER f
    #  2. DOUBLE epsilon
    #  3. DOUBLE a
    #  4. DOUBLE y_a
    #  5. DOUBLE b
    #
    def solveByRungeKutta(f, epsilon, a, y_a, b):
        func = Result.get_function(f)
        h = 0.1
        y = y_a
        while a < b:
            k_1 = func(a, y)
            k_2 = func(a + h/2, y + h/2 * k_1)
            k_3 = func(a + h/2, y + h/2 * k_2)
            k_4 = func(a + h, y + h * k_3)

            y_pred = y
            y = y + h/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

            error = abs(y - y_pred)
            if error > epsilon:
                h = h * error

            a = a + h
        return y

if __name__ == '__main__':

    f = int(input().strip())

    epsilon = float(input().strip())

    a = float(input().strip())

    y_a = float(input().strip())

    b = float(input().strip())

    result = Result.solveByRungeKutta(f, epsilon, a, y_a, b)

    print(str(result) + '\n')
