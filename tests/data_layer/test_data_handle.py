"""Test ModelData
"""
from unittest.mock import MagicMock, Mock, PropertyMock

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
        'narratives': {},
        'sos_model': 'test_sos_model'})
    store.write_sos_model({
        'name': 'test_sos_model',
        'sector_models': ['sector_model_test']})
    store._initial_conditions = {'sector_model_test': []}
    data = {'water_asset_a': {
                    'build_year': 2010,
                    'capacity': 50,
                    'location': None,
                    'sector': ''},
            'water_asset_b': {
                    'build_year': 2015,
                    'capacity': 150,
                    'location': None,
                    'sector': ''},
            'water_asset_c': {
                    'capacity': 100,
                    'build_year': 2015,
                    'location': None,
                    'sector': ''}}
    store._interventions['test_model'] = data
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

        A call to ``get_state`` method on the data handle calls the read_state
        method of the data_interface with arguments for model run name, current
        timestep and decision iteration.
        """
        mock_store.read_state = Mock(return_value=[
            {'name': 'test', 'build_year': 2010}])
        mock_store._interventions['test_model'] = [
            {'name': 'test',
             'capital_cost': {'value': 2500, 'unit': '£/GW'}
             }]
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = [{'name': 'test',
                     'build_year': 2010}]
        actual = data_handle.get_state()
        mock_store.read_state.assert_called_with(1, 2015, None)
        assert actual == expected

    def test_get_interventions_for_sector_model(self, mock_store, mock_model):
        """

        A call to the ``get_current_interventions`` method of the data_handle
        returns a dict of interventions from the current state filtered by the
        current model

        """
        state = [
            {'name': 'water_asset_a', 'build_year': 2010},
            {'name': 'water_asset_b', 'build_year': 2015}
        ]
        mock_store._state[(1, 2015, None)] = state
        # mock_store._strategies[1] = []

        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        actual = data_handle.get_current_interventions()
        expected = {
            'water_asset_a':
                {'build_year': 2010,
                 'capacity': 50,
                 'location': None,
                 'sector': ''},
            'water_asset_b':
                {'build_year': 2015,
                 'capacity': 150,
                 'location': None,
                 'sector': ''}
            }
        assert actual == expected

    def test_interventions_sector_model_ignore_unrecog(self, mock_store, mock_model):
        """Ignore unrecognised interventions

        Interventions that are listed in state, but not included in the
        intervention list are ignored

        """
        state = [
            {'name': 'water_asset_a', 'build_year': 2010},
            {'name': 'energy_asset_unexpected', 'build_year': 2015}
        ]

        mock_store._state[(1, 2015, None)] = state

        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        actual = data_handle.get_current_interventions()
        expected = {'water_asset_a': {
                'build_year': 2010,
                'capacity': 50,
                'location': None,
                'sector': ''
            }
        }
        assert actual == expected

    def test_pass_none_to_timestep_raises(self, mock_store, mock_model):
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


class TestDataHandleGetResults:

    @fixture(scope='function')
    def mock_sector_model(self):
        mock_sector_model = MagicMock()
        type(mock_sector_model).outputs = PropertyMock(return_value={'test_output': 'spec'})
        type(mock_sector_model).name = PropertyMock(return_value='test_sector_model')
        return mock_sector_model

    @fixture(scope='function')
    def mock_sos_model(self, mock_sector_model):
        mock_sos_model = MagicMock(outputs=[('test_sector_model', 'test_output')])
        mock_sos_model.name = 'test_sos_model'
        mock_sos_model.models = {'test_sector_model': mock_sector_model}
        return mock_sos_model

    def test_get_results_sectormodel(self, mock_store,
                                     mock_sector_model):
        """Get results from a sector model
        """
        store = mock_store
        store.write_results(42, 1, 'test_sector_model', 'spec', 2010, None, None)

        dh = DataHandle(mock_store, 1, 2010, [2010], mock_sector_model)
        actual = dh.get_results('test_output')
        expected = 42
        assert actual == expected

    def test_get_results_sos_model(self, mock_store,
                                   mock_sector_model,
                                   mock_sos_model):
        """Get results from a sector model within a sos model
        """
        store = mock_store
        store.write_results(42, 1, 'test_sector_model', 'spec', 2010, None, None)

        dh = DataHandle(mock_store, 1, 2010, [2010], mock_sos_model)
        actual = dh.get_results('test_output',
                                model_name='test_sector_model')
        expected = 42
        assert actual == expected

    def test_get_results_no_output_sos(self, mock_store,
                                       mock_sos_model):
        with raises(KeyError):
            dh = DataHandle(mock_store, 1, 2010, [2010], mock_sos_model)
            dh.get_results('no_such_output',
                           model_name='test_sector_model')

    def test_get_results_no_output_sector(self, mock_store,
                                          mock_sector_model):

        with raises(KeyError):
            dh = DataHandle(mock_store, 1, 2010, [2010], mock_sector_model)
            dh.get_results('no_such_output')

    def test_get_results_wrong_name_sos(self,
                                        mock_store,
                                        mock_sos_model):
        with raises(KeyError):
            dh = DataHandle(mock_store, 1, 2010, [2010], mock_sos_model)
            dh.get_results('test_output',
                           model_name='no_such_model')
