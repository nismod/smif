"""Tests the ModelOutputs class
"""
from numpy.testing import assert_equal
from smif.outputs import ModelOutputs


class TestModelOutputs:
    def test_static_result_parsing(self):
        results = [{'capacity': 5.0, 'cost': 6.32, 'water': 3.0}]
        mo = ModelOutputs(results)
        actual = mo.outputs.names
        expected = ['capacity', 'cost', 'water']
        # assert actual == expected
        actual = mo.outputs.values
        expected = [[5.0], [6.32], [3.0]]
        assert_equal(actual, expected)

    def test_dynamic_result_parsing(self):
        results = [{'capacity': 5.0, 'cost': 6.32, 'water': 3.0},
                   {'capacity': 5.0, 'cost': 0.0, 'water': 3.0},
                   {'capacity': 5.0, 'cost': 0.0, 'water': 3.0}]
        mo = ModelOutputs(results)
        actual = mo.outputs.names
        expected = ['capacity', 'cost', 'water']
        # assert actual == expected
        actual = mo.outputs.values
        expected = [[5.0, 5.0, 5.0], [6.32, 0.0, 0.0], [3.0, 3.0, 3.0]]
        assert_equal(actual, expected)
