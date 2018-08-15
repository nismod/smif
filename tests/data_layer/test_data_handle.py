"""Test ModelData
"""
from unittest.mock import MagicMock, Mock

import numpy as np
from pytest import fixture, raises
from smif.data_layer import DataHandle
from smif.data_layer.data_handle import TimestepResolutionError
from smif.metadata import Spec
from smif.model import SectorModel


@fixture(scope='function')
def mock_store():
    store = Mock()
    store.read_results = MagicMock(return_value=np.array([[1.0]]))
    return store


class EmptySectorModel(SectorModel):
    """Sector Model implementation
    """
    def simulate(self, data_handle):
        return data_handle


@fixture(scope='function')
def mock_model():
    model = EmptySectorModel('test_model')
    model.add_parameter(
        Spec(
            name='smart_meter_savings',
            description='The savings from smart meters',
            abs_range=(0, 100),
            exp_range=(3, 10),
            default=3,
            unit='%',
            dtype='float'
        )
    )

    spec = Spec(
        name='test',
        dtype='float',
        dims=['region', 'interval'],
        coords={'region': [1, 2], 'interval': ['a', 'b']}
    )
    model.add_input(spec)
    model.add_output(spec)

    source_model = EmptySectorModel('test_source')
    source_model.add_output(spec)
    model.add_dependency(source_model, 'test', 'test')

    return model


@fixture(scope='function')
def mock_model_with_conversion():
    model = EmptySectorModel('test_model')
    model.add_parameter(
        Spec(
            name='smart_meter_savings',
            description='The savings from smart meters',
            abs_range=(0, 100),
            exp_range=(3, 10),
            default=3,
            unit='%',
            dtype='float'
        )
    )

    spec = Spec(
        name='test',
        dtype='float',
        dims=['half_squares', 'remap_months'],
        coords={'half_squares': [1, 2], 'remap_months': ['jan', 'feb']},
        unit='ml'  # millilitres to convert
    )
    model.add_input(spec)
    model.add_output(spec)

    source_model = EmptySectorModel('test_source')
    output_spec = Spec(
        name='test',
        dtype='float',
        dims=['half_squares', 'remap_months'],
        coords={'half_squares': [1, 2], 'remap_months': ['jan', 'feb']},
        unit='l'  # litres convert to
    )
    source_model.add_output(output_spec)
    model.add_dependency(source_model, 'test', 'test')

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
            mock_model.inputs['test'],  # input spec
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
            mock_model_with_conversion.inputs['test'],
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
            mock_model.inputs['test'],
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
            mock_model.inputs['test'],
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
            mock_model.inputs['test'],
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
            mock_model.inputs['test'],
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
            mock_model.inputs['test'],
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
