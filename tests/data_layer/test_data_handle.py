"""Test ModelData
"""
from unittest.mock import MagicMock, Mock

import numpy as np
from pytest import fixture, raises
from smif.data_layer import DataHandle
from smif.data_layer.data_handle import TimestepResolutionError
from smif.metadata import Metadata, MetadataSet
from smif.model.dependency import Dependency
from smif.parameters import ParameterList


@fixture(scope='function')
def mock_store():
    store = Mock()
    store.read_results = MagicMock(return_value=np.array([[1.0]]))
    return store


@fixture(scope='function')
def mock_model():
    model = Mock()
    model.name = 'test_model'
    model.parameters = ParameterList()
    model.parameters.add_parameter(
        {
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%',
            'parent': None
        }
    )
    regions = Mock()
    regions.name = 'half_squares'
    regions.__len__ = lambda self: 1
    intervals = Mock()
    intervals.name = 'remap_months'
    intervals.__len__ = lambda self: 1
    model.inputs = MetadataSet([
        Metadata('test', regions, intervals, 'm')
    ])
    model.outputs = MetadataSet([
        Metadata('test', regions, intervals, 'm')
    ])
    source_model = Mock()
    source_model.name = 'test_source'
    model.deps = {
        'test': Dependency(
            source_model,
            Metadata('test_output', regions, intervals, 'm'),
            Metadata('test', regions, intervals, 'm')
        )
    }
    return model


@fixture(scope='function')
def mock_model_with_conversion():
    model = Mock()
    model.name = 'test_model'
    model.parameters = ParameterList()
    model.parameters.add_parameter(
        {
            'name': 'smart_meter_savings',
            'description': 'The savings from smart meters',
            'absolute_range': (0, 100),
            'suggested_range': (3, 10),
            'default_value': 3,
            'units': '%',
            'parent': None
        }
    )
    regions = Mock()
    regions.name = 'half_squares'
    intervals = Mock()
    intervals.name = 'remap_months'
    model.inputs = MetadataSet([
        Metadata('test', regions, intervals, 'liters')
    ])
    source_model = Mock()
    source_model.name = 'test_source'
    model.deps = {
        'test': Dependency(
            source_model,
            Metadata('test_output', regions, intervals, 'milliliters'),
            Metadata('test', regions, intervals, 'liters')
        )
    }
    return model


class TestDataHandle():
    def test_create(self):
        """should be created with a DataInterface
        """
        DataHandle(Mock(), 1, 2015, [2015, 2020], Mock())

    def test_get_data(self, mock_store, mock_model):
        """should allow read access to input data
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = np.array([[1.0]])
        actual = data_handle.get_data("test")
        assert actual == expected

        mock_store.read_results.assert_called_with(
            1,
            'test_source',  # read from source model
            'test_output',  # using source model output name
            'half_squares',
            'remap_months',
            2015,
            None,
            None
        )

    def test_get_data_with_conversion(self, mock_store, mock_model_with_conversion):
        """should convert liters to milliliters (1 -> 0.001)
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model_with_conversion)
        expected = np.array([[0.001]])
        actual = data_handle.get_data("test")
        assert actual == expected

        mock_store.read_results.assert_called_with(
            1,
            'test_source',  # read from source model
            'test_output',  # using source model output name
            'half_squares',
            'remap_months',
            2015,
            None,
            None
        )

    def test_get_base_timestep_data(self, mock_store, mock_model):
        """should allow read access to input data from base timestep
        """
        data_handle = DataHandle(mock_store, 1, 2025, [2015, 2020, 2025], mock_model)
        expected = np.array([[1.0]])
        actual = data_handle.get_base_timestep_data("test")
        assert actual == expected

        mock_store.read_results.assert_called_with(
            1,
            'test_source',  # read from source model
            'test_output',  # using source model output name
            'half_squares',
            'remap_months',
            2015,  # base timetep
            None,
            None
        )

    def test_get_previous_timestep_data(self, mock_store, mock_model):
        """should allow read access to input data from previous timestep
        """
        data_handle = DataHandle(mock_store, 1, 2025, [2015, 2020, 2025], mock_model)
        expected = np.array([[1.0]])
        actual = data_handle.get_previous_timestep_data("test")
        assert actual == expected

        mock_store.read_results.assert_called_with(
            1,
            'test_source',  # read from source model
            'test_output',  # using source model output name
            'half_squares',
            'remap_months',
            2020,  # previous timetep
            None,
            None
        )

    def test_get_data_with_square_brackets(self, mock_store, mock_model):
        """should allow dict-like read access to input data
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = np.array([[1.0]])
        actual = data_handle["test"]
        assert actual == expected

        mock_store.read_results.assert_called_with(
            1,
            'test_source',  # read from source model
            'test_output',  # using source model output name
            'half_squares',
            'remap_months',
            2015,
            None,
            None
        )

    def test_set_data(self, mock_store, mock_model):
        """should allow write access to output data
        """
        expected = np.array([[1.0]])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        data_handle.set_results("test", expected)
        assert data_handle["test"] == expected

        mock_store.write_results.assert_called_with(
            1,
            'test_model',
            'test',
            expected,
            'half_squares',
            'remap_months',
            2015,
            None,
            None
        )

    def test_set_data_wrong_shape(self, mock_store, mock_model):
        """should allow write access to output data
        """
        expect_error = np.array([[1.0, 1.0]])  # regions is 1, intervals is 1 not 2
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        with raises(ValueError) as ex:
            data_handle.set_results("test", expect_error)

        assert "Tried to set results with shape (1, 2), " + \
            "expected (1, 1) for test_model:test" in str(ex)

    def test_set_data_with_square_brackets(self, mock_store, mock_model):
        """should allow dict-like write access to output data
        """
        expected = np.array([[1.0]])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        data_handle["test"] = expected
        assert data_handle["test"] == expected

        mock_store.write_results.assert_called_with(
            1,
            'test_model',
            'test',
            expected,
            'half_squares',
            'remap_months',
            2015,
            None,
            None
        )

    def test_get_regions(self, mock_store, mock_model):
        """should allow read access to input data
        """
        mock_store.read_region_names = Mock(return_value=['a', 'b'])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = ['a', 'b']
        actual = data_handle.get_region_names("half_squares")
        assert actual == expected

        mock_store.read_region_names.assert_called_with(
            'half_squares')

    def test_get_intervals(self, mock_store, mock_model):
        """should allow read access to input data
        """
        mock_store.read_interval_names = Mock(return_value=['a', 'b'])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = ['a', 'b']
        actual = data_handle.get_interval_names("remap_months")
        assert actual == expected

        mock_store.read_interval_names.assert_called_with(
            'remap_months')


class TestDataHandleState():
    """Test handling of initial conditions, decision interventions and intervention state.
    """
    def test_get_state(self, mock_store, mock_model):
        """should get decision module state for given timestep/decision_iteration
        """
        mock_store.read_state = Mock(return_value=[('test', 2010)])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = [('test', 2010)]
        actual = data_handle.get_state()
        assert actual == expected

    def test_get_state_raises_when_timestep_is_none(self, mock_store, mock_model):

        data_handle = DataHandle(mock_store, 1, None, [2015, 2020], mock_model)
        with raises(ValueError):
            data_handle.get_state()


class TestDataHandleTimesteps():
    """Test timestep helper properties
    """
    def test_current_timestep(self):
        """should return current timestep
        """
        data_handle = DataHandle(Mock(), 1, 2015, [2015, 2020], Mock())
        assert data_handle.current_timestep == 2015

    def test_base_timestep(self):
        """should return first timestep in list
        """
        data_handle = DataHandle(Mock(), 1, 2015, [2015, 2020], Mock())
        assert data_handle.base_timestep == 2015

    def test_previous_timestep(self):
        """should return previous timestep from list
        """
        data_handle = DataHandle(Mock(), 1, 2020, [2015, 2020], Mock())
        assert data_handle.previous_timestep == 2015

    def test_previous_timestep_error(self):
        """should raise error if there's no previous timestep in the list
        """
        data_handle = DataHandle(Mock(), 1, 2015, [2015, 2020], Mock())
        with raises(TimestepResolutionError) as ex:
            data_handle.previous_timestep
        assert 'no previous timestep' in str(ex)
