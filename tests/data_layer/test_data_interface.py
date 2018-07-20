"""Test data interface
"""
import numpy as np
from pytest import raises
from smif.data_layer import DataMismatchError
from smif.data_layer.data_interface import DataInterface


class TestDataInterface():
    def test_data_list_to_array(self):

        data = [
            {
                'year': 2010,
                'value': 3,
                'region': 'oxford',
                'interval': '1'
            }
        ]
        actual = DataInterface.data_list_to_ndarray(
            data,
            ['oxford'],
            ['1']
        )
        expected = np.array([[3.]], dtype=float)
        np.testing.assert_equal(actual, expected)

    def test_scenario_data_missing_year(self, oxford_region):
        data = [
            {
                'value': 3.14
            }
        ]
        msg = "Observation missing region"
        with raises(KeyError) as ex:
            DataInterface.data_list_to_ndarray(
                data,
                ['oxford'],
                ['1']
            )
        assert msg in str(ex.value)

    def test_scenario_data_missing_param_region(self, oxford_region):
        data = [
            {
                'value': 3.14,
                'region': 'missing',
                'interval': '1',
                'year': 2015
            }
        ]
        msg = "Unknown region 'missing' in row 0"
        with raises(ValueError) as ex:
            DataInterface.data_list_to_ndarray(
                data,
                ['oxford'],
                ['1']
            )
        assert msg in str(ex)

    def test_scenario_data_missing_param_interval(self):
        data = [
            {
                'value': 3.14,
                'region': 'oxford',
                'interval': '1',
                'year': 2015
            },
            {
                'value': 3.14,
                'region': 'oxford',
                'interval': 'extra',
                'year': 2015
            }
        ]
        msg = "Number of observations (2) is not equal to intervals (1) x regions (1)"
        with raises(DataMismatchError) as ex:
            DataInterface.data_list_to_ndarray(
                data,
                ['oxford'],
                ['1']
            )
        assert msg in str(ex)
