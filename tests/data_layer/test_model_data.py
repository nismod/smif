"""Test ModelData
"""
from unittest.mock import Mock

import numpy as np
from smif.data_layer.model_data import DataHandle


def test_create():
    """should be created with a DataInterface
    """
    DataHandle(Mock(), 1)


def test_get_data():
    """should allow read access to input data
    """
    expected = np.array([[1.0]])
    md = DataHandle({
        "test": expected
    }, 1)

    actual = md["test"]
    assert actual == expected

    actual = md.get_data("test")
    assert actual == expected


def test_set_data():
    """should allow write access to output data
    """
    expected = np.array([[1.0]])
    md = DataHandle({}, 1)

    md["test"] = expected
    assert md["test"] == expected

    md.set_results("test", expected)
    assert md["test"] == expected
