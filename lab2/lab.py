import math
import os
import random
import re
import sys

class Result:
    isMethodApplicable = True
    errorMessage = ""
    #
    # Complete the 'solveByGaussSeidel' function below.
    #
    # The function is expected to return a DOUBLE_ARRAY.
    # The function accepts following parameters:
    #  1. INTEGER n
    #  2. 2D_DOUBLE_ARRAY matrix
    #  3. INTEGER epsilon
    #

    def swap_str(matrix):
        size_matrix = len(matrix)
        max_value = 0
        save_j = 0
        for i in range(size_matrix):
            for j in range(i, size_matrix):
                if abs(matrix[i][j]) > max_value:
                    max_value = abs(matrix[i][j])
                    save_j = j
            max_value = 0
            for k in range(size_matrix):
                matrix[k][i], matrix[k][save_j] = matrix[k][save_j], matrix[k][i]

    def swap_column(matrix):
        size_matrix = len(matrix)
        max_value = 0
        save_j = 0
        for i in range(size_matrix):
            for j in range(i, size_matrix):
                if abs(matrix[j][i]) > max_value:
                    max_value = abs(matrix[j][i])
                    save_j = j
            max_value = 0
            for k in range(size_matrix):
                matrix[i][k], matrix[save_j][k] = matrix[save_j][k], matrix[i][k]

    def is_diagonal_dominate(matrix) -> bool:
        size_matrix = len(matrix)
        for i in range(size_matrix):
            sums = sum(abs(matrix[i][j]) for j in range(size_matrix) if j != i)
            if sums > abs(matrix[i][i]) or matrix[i][i] == 0: return False
        return True

    def solveByGaussSeidel(n, matrix, epsilon):
        if Result.is_diagonal_dominate(matrix=matrix) == False:
            Result.swap_str(matrix)
            if Result.is_diagonal_dominate(matrix=matrix) == False:
                Result.swap_column(matrix)
                if Result.is_diagonal_dominate(matrix=matrix) == False:
                    Result.isMethodApplicable = False
                    Result.errorMessage = "The system has no diagonal dominance for this method. Method of the Gauss-Seidel is not applicable"
                    return
        
        vectox_x_before = [0] * n
        vectox_x_after = [0] * n
        for m in range(10000):
            for i in range(n):
                vectox_x_after[i] = (matrix[i][n] - sum(matrix[i][j] * vectox_x_after[j] for j in range(n) if j != i)) / matrix[i][i]
            max_err = max(abs(vectox_x_after[n] - vectox_x_before[n]) for n in range(n))
            if max_err <= epsilon:
                return vectox_x_after
            vectox_x_before = vectox_x_after.copy()


if __name__ == '__main__':
    n = int(input().strip())

    matrix_rows = n
    matrix_columns = n+1

    matrix = []

    for _ in range(matrix_rows):
        matrix.append(list(map(float, input().rstrip().split())))

    epsilon = float(input().strip())

    result = Result.solveByGaussSeidel(n, matrix, epsilon)
    if Result.isMethodApplicable:
        print('\n'.join(map(str, result)))
    else:
        print(f"{Result.errorMessage}")