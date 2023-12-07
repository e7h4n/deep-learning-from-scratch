from ml.chapter_2 import AND

def test_2_3_1():
    assert AND(0, 0) == 0
    assert AND(1, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 1) == 1
