import pytest
import numpy as np
import pprint
import matplotlib.pylab as plt
from ml.chapter_3 import init_network, relu, sigmoid, step_function, forward

pp = pprint.PrettyPrinter(indent=4)
p = pp.pprint


def test_3_2_2():
    x = np.array([-1.0, 1.0, 2.0])
    p(x)
    y = x > 0
    p(y)
    z = y.astype(int)
    p(z)


@pytest.mark.skip(reason="only should be run manually")
def test_3_2_3():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


@pytest.mark.skip(reason="only should be run manually")
def test_3_2_4():
    x = np.array([-1.0, 1.0, 2.0])
    p(sigmoid(x))

    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


@pytest.mark.skip(reason="only should be run manually")
def test_3_2_7():
    x = np.array([-1.0, 1.0, 2.0])
    p(sigmoid(x))

    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def test_3_3_1():
    A = np.array([1, 2, 3, 4])
    p(A)
    p(np.ndim(A))
    p(A.shape)
    p(A.shape[0])

    B = np.array([[1, 2], [3, 4], [5, 6]])
    p(B)
    p(np.ndim(B))
    p(B.shape)


def test_3_3_2():
    A = np.array([[1, 2], [3, 4]])
    p(A.shape)
    B = np.array([[5, 6], [7, 8]])
    p(B.shape)
    p(np.dot(A, B))

    A = np.array([[1, 2, 3], [4, 5, 6]])
    p(A.shape)
    B = np.array([[1, 2], [3, 4], [5, 6]])
    p(B.shape)
    p(np.dot(A, B))

    C = np.array([[1, 2], [3, 4]])
    p(C.shape)
    p(A.shape)

    A = np.array([[1, 2], [3, 4], [5, 6]])
    p(A.shape)
    B = np.array([7, 8])
    p(B.shape)
    p(np.dot(A, B))


def test_3_3_3():
    X = np.array([1, 2])
    p(X.shape)
    W = np.array([[1, 3, 5], [2, 4, 6]])
    p(W)
    p(W.shape)
    Y = np.dot(X, W)
    p(Y)


def test_3_4_2():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    p(W1.shape)
    p(X.shape)
    p(B1.shape)

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    p(A1)
    p(Z1)

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    p(Z1.shape)
    p(W2.shape)
    p(B2.shape)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    p(A2)
    p(Z2)

    def identity_function(x):
        return x

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)

    p(Y)


def test_3_4_3():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    p(y)
