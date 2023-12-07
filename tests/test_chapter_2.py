from ml.chapter_2 import AND, NAND, OR


def test_2_3_1():
    assert AND(0, 0) == 0
    assert AND(1, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 1) == 1


def test_2_3_2():
    import numpy as np

    x = np.array([0, 1])
    w = np.array([0.5, 0.5])
    b = -0.7
    print(w * x)
    print(np.sum(w * x))
    print(np.sum(w * x) + b)


def test_2_3_3():
    assert NAND(0, 0) == 1
    assert NAND(1, 0) == 1
    assert NAND(0, 1) == 1
    assert NAND(1, 1) == 0

    assert OR(0, 0) == 0
    assert OR(1, 0) == 1
    assert OR(0, 1) == 1
    assert OR(1, 1) == 1
