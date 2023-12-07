import sys, os

sys.path.append("./author-project")
sys.path.append("./author-project/ch04")
import pytest
import numpy as np
from pprint import pprint
import matplotlib.pylab as plt
from practice.chapter_3 import init_network, relu, sigmoid, step_function, forward
from practice.chapter_4 import simpleNet
from common.functions import *
from common.gradient import numerical_gradient
from two_layer_net import TwoLayerNet


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def test_4_3_2():
    def function_1(x):
        return 0.01 * x**2 + 0.1 * x

    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    # plt.show()

    pprint(numerical_diff(function_1, 5))
    pprint(numerical_diff(function_1, 10))


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.shape[0]):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def test_4_4():
    def function_2(x):
        return x[0] ** 2 + x[1] ** 2

    pprint(numerical_gradient(function_2, np.array([3.0, 4.0])))
    pprint(numerical_gradient(function_2, np.array([0, 2.0])))
    pprint(numerical_gradient(function_2, np.array([3.0, 0.0])))

    def gradient_descent(f, init_x, lr=0.01, step_num=100):
        x = init_x

        for i in range(step_num):
            grad = numerical_gradient(f, x)
            x -= lr * grad

        return x

    init_x = np.array([-3.0, 4.0])
    pprint(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

    init_x = np.array([-3.0, 4.0])
    pprint(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))

    init_x = np.array([-3.0, 4.0])
    pprint(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))


def test_4_4_2():
    def numerical_gradient(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 値を元に戻す
            it.iternext()

        return grad

    net = simpleNet()
    pprint(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    t = np.array([0, 0, 1])

    f = lambda w: net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    pprint(dW)


def test_4_5_1():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    pprint(net.params["W1"].shape)
    pprint(net.params["b1"].shape)
    pprint(net.params["W2"].shape)
    pprint(net.params["b2"].shape)

    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    pprint(net.accuracy(x, t))

    grads = net.numerical_gradient(x, t)
    pprint(grads["W1"].shape)
    pprint(grads["b1"].shape)
    pprint(grads["W2"].shape)
    pprint(grads["b2"].shape)
