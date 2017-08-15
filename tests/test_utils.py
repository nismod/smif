"""Test utility code
"""
from smif.utils import Singleton


class SingletonClass(object, metaclass=Singleton):
    def __init__(self):
        self.data = []


class SingletonSubClass(SingletonClass):
    pass


def test_singleton_creation():
    sc_1 = SingletonClass()
    sc_2 = SingletonClass()
    assert sc_1 is sc_2


def test_singleton_inheritance():
    sc_1 = SingletonSubClass()
    sc_2 = SingletonSubClass()
    assert sc_1 is sc_2


def test_singleton_data_is_shared():
    sc_1 = SingletonClass()
    sc_2 = SingletonClass()
    sc_1.data = [1, 2]

    assert sc_1.data == [1, 2]
    assert sc_2.data == [1, 2]

    sc_2.data.append(3)
    assert sc_1.data[2] == 3
