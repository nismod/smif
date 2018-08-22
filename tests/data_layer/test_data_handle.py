"""Test ModelData
"""
from unittest.mock import Mock

import numpy as np
from pytest import fixture, raises
from smif.data_layer import DataHandle, MemoryInterface
from smif.data_layer.data_handle import TimestepResolutionError
from smif.metadata import Spec
from smif.model import SectorModel


@fixture(scope='function')
def mock_store():
    """Store with minimal setup
    """
    store = MemoryInterface()
    store.write_model_run({
        'name': 1,
        'narratives': {}
    })
    return store


class EmptySectorModel(SectorModel):
    """Sector Model implementation
    """
    def simulate(self, data):
        """no-op
        """
        return data


@fixture(scope='function')
def empty_model():
    """Minimal sector model
    """
    return EmptySectorModel('test_model')


@fixture(scope='function')
def mock_model():
    """Sector model with parameter, input, output, dependency
    """
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
    """Sector model with dependency via convertor
    """
    source = EmptySectorModel('test_source')
    convertor = EmptySectorModel('test_convertor')
    model = EmptySectorModel('test_model')

    ml_spec = Spec(
        name='test',
        dtype='float',
        dims=['half_squares', 'remap_months'],
        coords={'half_squares': [1, 2], 'remap_months': ['jan', 'feb']},
        unit='ml'  # millilitres to convert
    )
    l_spec = Spec(
        name='test',
        dtype='float',
        dims=['half_squares', 'remap_months'],
        coords={'half_squares': [1, 2], 'remap_months': ['jan', 'feb']},
        unit='l'  # litres convert to
    )
    source.add_output(ml_spec)
    convertor.add_input(ml_spec)
    convertor.add_output(l_spec)
    model.add_input(l_spec)

    convertor.add_dependency(source, 'test', 'test')
    model.add_dependency(convertor, 'test', 'test')

    return model


class TestDataHandle():
    """
    """
    def test_create(self, mock_model, mock_store):
        """should be created with a DataInterface
        """
        DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

    def test_get_data(self, mock_store, mock_model):
        """should allow read access to input data
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])

        mock_store.write_results(
            expected,
            1,
            'test_source',  # write source model results
            mock_model.inputs['test'],  # input spec must be equivalent
            2015,
            None,
            None
        )
        actual = data_handle.get_data("test")
        np.testing.assert_equal(actual, expected)

    def test_get_data_with_conversion(self, mock_store, mock_model_with_conversion):
        """should convert liters to milliliters (1 -> 0.001)
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model_with_conversion)
        expected = np.array([[0.001]])
        mock_store.write_results(
            expected,
            1,
            'test_convertor',  # write results as though from convertor
            mock_model_with_conversion.inputs['test'],
            2015,
            None,
            None
        )
        actual = data_handle.get_data("test")
        np.testing.assert_equal(actual, expected)

    def test_get_base_timestep_data(self, mock_store, mock_model):
        """should allow read access to input data from base timestep
        """
        data_handle = DataHandle(mock_store, 1, 2025, [2015, 2020, 2025], mock_model)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_store.write_results(
            expected,
            1,
            'test_source',  # write source model results
            mock_model.inputs['test'],
            2015,  # base timetep
            None,
            None
        )
        actual = data_handle.get_base_timestep_data("test")
        np.testing.assert_equal(actual, expected)

    def test_get_previous_timestep_data(self, mock_store, mock_model):
        """should allow read access to input data from previous timestep
        """
        data_handle = DataHandle(mock_store, 1, 2025, [2015, 2020, 2025], mock_model)
        expected = np.random.rand(*mock_model.inputs['test'].shape)
        mock_store.write_results(
            expected,
            1,
            'test_source',  # write source model results
            mock_model.inputs['test'],
            2020,  # previous timetep
            None,
            None
        )
        actual = data_handle.get_previous_timestep_data("test")
        np.testing.assert_equal(actual, expected)

    def test_get_data_with_square_brackets(self, mock_store, mock_model):
        """should allow dict-like read access to input data
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = np.random.rand(*mock_model.inputs['test'].shape)
        mock_store.write_results(
            expected,
            1,
            'test_source',  # write source model results
            mock_model.inputs['test'],
            2015,  # current timetep
            None,
            None
        )
        actual = data_handle["test"]
        np.testing.assert_equal(actual, expected)

    def test_set_data(self, mock_store, mock_model):
        """should allow write access to output data
        """
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        data_handle.set_results("test", expected)
        actual = mock_store.read_results(
            1,
            'test_model',  # read results from model
            mock_model.outputs['test'],
            2015,
            None,
            None
        )
        np.testing.assert_equal(actual, expected)

    def test_set_data_wrong_shape(self, mock_store, mock_model):
        """should allow write access to output data
        """
        expect_error = np.array([[1.0, 1.0]])  # regions is 1, intervals is 1 not 2
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        with raises(ValueError) as ex:
            data_handle.set_results("test", expect_error)

        msg = "Tried to set results with shape (1, 2), expected (2, 2) for test_model:test"
        assert msg in str(ex)

    def test_set_data_with_square_brackets(self, mock_store, mock_model):
        """should allow dict-like write access to output data
        """
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        data_handle["test"] = expected
        actual = mock_store.read_results(
            1,
            'test_model',  # read results from model
            mock_model.outputs['test'],
            2015,
            None,
            None
        )
        np.testing.assert_equal(actual, expected)

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
    def test_current_timestep(self, empty_model, mock_store):
        """should return current timestep
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], empty_model)
        assert data_handle.current_timestep == 2015

    def test_base_timestep(self, empty_model, mock_store):
        """should return first timestep in list
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], empty_model)
        assert data_handle.base_timestep == 2015

    def test_previous_timestep(self, empty_model, mock_store):
        """should return previous timestep from list
        """
        data_handle = DataHandle(mock_store, 1, 2020, [2015, 2020], empty_model)
        assert data_handle.previous_timestep == 2015

    def test_previous_timestep_error(self, empty_model, mock_store):
        """should raise error if there's no previous timestep in the list
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], empty_model)
        with raises(TimestepResolutionError) as ex:
            data_handle.previous_timestep
        assert 'no previous timestep' in str(ex)
