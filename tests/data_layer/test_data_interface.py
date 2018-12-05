"""Test data interface classmethods
"""
import numpy as np
from pytest import raises
from smif.data_layer.data_array import DataArray
from smif.data_layer.datafile_interface import (data_list_to_ndarray,
                                                ndarray_to_data_list)
from smif.exception import SmifDataMismatchError
from smif.metadata import Coordinates, Spec


def test_data_list_to_array():
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

    actual = data_list_to_ndarray(
        data,
        spec
    )
    expected = DataArray(spec, np.array([[3.]], dtype=float))

    assert actual == expected


def test_ndarray_to_data_list():

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
        coords=[Coordinates('region', ['oxford']),
                Coordinates('interval', ['1'])],
        dtype='int')

    da = DataArray(spec, np.array([[3]]))

    actual = ndarray_to_data_list(da, timestep=2010)
    expected = data

    assert actual == expected


def test_ndarray_with_spatial_dimension(sample_dimensions):
    lad = sample_dimensions[0]['elements']
    spec = Spec(
        name='test',
        coords=[Coordinates('lad', lad), Coordinates('interval', ['1'])],
        dtype='int'
    )
    data = np.array([[42], [69]])

    print(spec.coords)

    da = DataArray(spec, data)

    actual = ndarray_to_data_list(da, timestep=2010)
    expected = [
        {
            'timestep': 2010,
            'test': 42,
            'lad': 'a',
            'interval': '1'
        },
        {
            'timestep': 2010,
            'test': 69,
            'lad': 'b',
            'interval': '1'
        },
    ]
    assert actual == expected


def test_scenario_data_missing_timestep():
    data = [
        {
            'test': 3.14
        }
    ]
    msg = "Observation missing dimension key (region)"
    with raises(KeyError) as ex:
        data_list_to_ndarray(
            data,
            Spec(
                name='test',
                dims=['region', 'interval'],
                coords={'region': ['oxford'], 'interval': ['1']},
                dtype='int'
            )
        )
    assert msg in str(ex.value)


def test_scenario_data_missing_param_region():
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
        data_list_to_ndarray(
            data,
            Spec(
                name='test',
                dims=['region', 'interval'],
                coords={'region': ['oxford'], 'interval': ['1']},
                dtype='int'
            )
        )
    assert msg in str(ex)


def test_scenario_data_missing_param_interval():
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
        data_list_to_ndarray(
            data,
            Spec(
                name='test',
                dims=['region', 'interval'],
                coords={'region': ['oxford'], 'interval': ['1']},
                dtype='int'
            )
        )
    assert msg in str(ex)
