"""Test data interface classmethods
"""
import numpy as np
from pytest import raises
from smif.data_layer.data_array import DataArray
from smif.data_layer.data_interface import DataInterface
from smif.exception import SmifDataMismatchError
from smif.metadata import Spec


class TestDataInterface():
    def test_data_list_to_array(self):
        data = [
            {
                'timestep': 2010,
                'test': 3,
                'region': 'oxford',
                'interval': '1'
            }
        ]

        spec = Spec(
                name='test',
                dims=['region', 'interval'],
                coords={'region': ['oxford'], 'interval': ['1']},
                dtype='int'
                )

        actual = DataInterface.data_list_to_ndarray(
            data,
            spec
        )
        expected = DataArray(spec, np.array([[3.]], dtype=float))

        assert actual == expected

    def test_ndarray_to_data_list(self):

        data = [
            {
                'timestep': 2010,
                'test': 3,
                'region': 'oxford',
                'interval': '1'
            }
        ]

        spec = Spec(
                name='test',
                dims=['region', 'interval'],
                coords={'region': ['oxford'], 'interval': ['1']},
                dtype='int'
                )

        actual = DataInterface.ndarray_to_data_list(
            np.array([[3]]),
            spec, timestep=2010
        )
        expected = data

        assert actual == expected

    def test_scenario_data_missing_timestep(self):
        data = [
            {
                'test': 3.14
            }
        ]
        msg = "Observation missing dimension key (region)"
        with raises(KeyError) as ex:
            DataInterface.data_list_to_ndarray(
                data,
                Spec(
                    name='test',
                    dims=['region', 'interval'],
                    coords={'region': ['oxford'], 'interval': ['1']},
                    dtype='int'
                )
            )
        assert msg in str(ex.value)

    def test_scenario_data_missing_param_region(self):
        data = [
            {
                'test': 3.14,
                'region': 'missing',
                'interval': '1',
                'timestep': 2015
            }
        ]
        msg = "Unknown region 'missing' in row 0"
        with raises(ValueError) as ex:
            DataInterface.data_list_to_ndarray(
                data,
                Spec(
                    name='test',
                    dims=['region', 'interval'],
                    coords={'region': ['oxford'], 'interval': ['1']},
                    dtype='int'
                )
            )
        assert msg in str(ex)

    def test_scenario_data_missing_param_interval(self):
        data = [
            {
                'test': 3.14,
                'region': 'oxford',
                'interval': '1',
                'timestep': 2015
            },
            {
                'test': 3.14,
                'region': 'oxford',
                'interval': 'extra',
                'timestep': 2015
            }
        ]
        msg = "Number of observations (2) is not equal to product of (1, 1)"
        with raises(SmifDataMismatchError) as ex:
            DataInterface.data_list_to_ndarray(
                data,
                Spec(
                    name='test',
                    dims=['region', 'interval'],
                    coords={'region': ['oxford'], 'interval': ['1']},
                    dtype='int'
                )
            )
        assert msg in str(ex)
