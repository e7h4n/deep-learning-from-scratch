import numpy as np
import matplotlib.pylab as plt


def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function_2(x):
    y = x > 0
    return y.astype(int)


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)
