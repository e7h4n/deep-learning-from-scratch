import pytest
import numpy as np
import pprint
import matplotlib.pylab as plt
from ml.chapter_3 import relu, sigmoid, step_function

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
