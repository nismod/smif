"""Test ModelData
"""
# pylint: disable=redefined-outer-name
from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
from pytest import fixture, raises
from smif.data_layer import DataHandle
from smif.data_layer.data_array import DataArray
from smif.data_layer.data_handle import ResultsHandle
from smif.exception import (SmifDataError, SmifDataMismatchError,
                            SmifTimestepResolutionError)
from smif.metadata import Spec
from smif.model import SectorModel


@fixture(scope='function')
def mock_store(sample_dimensions, get_sector_model, empty_store):
    """Store with minimal setup
    """
    store = empty_store

    for dim in sample_dimensions:
        store.write_dimension(dim)

    store.write_model_run({
        'name': 1,
        'narratives': {},
        'sos_model': 'test_sos_model',
        'scenarios': {}
    })
    store.write_sos_model({
        'name': 'test_sos_model',
        'sector_models': ['energy_demand'],
        'scenario_dependencies': [],
        'model_dependencies': [
            {
                'source': 'test_source',
                'source_output': 'test',
                'sink_input': 'test',
                'sink': 'energy_demand'
            }
        ],
        'narratives': [
            {
                'name': 'test_narrative',
                'description': 'a narrative config',
                'provides': {
                    'energy_demand': ['smart_meter_savings']
                },
                'variants': [
                    {
                        'name': 'high_tech_dsm',
                        'description': 'High takeup',
                        'data': {'smart_meter_savings': 'filename.csv'}
                    }
                ]
            }
        ]
    })
    parameter_spec = Spec(
        name='smart_meter_savings',
        dtype='float',
        unit='%'
    )
    da = DataArray(parameter_spec, np.array(99))
    store.write_narrative_variant_data(
        'test_sos_model', 'test_narrative', 'high_tech_dsm', da)

    store.write_model_run({
        'name': 2,
        'narratives': {},
        'sos_model': 'test_converting_sos_model',
        'scenarios': {}
    })
    store.write_sos_model({
        'name': 'test_converting_sos_model',
        'sector_models': ['energy_demand'],
        'scenario_dependencies': [],
        'model_dependencies': [
            {
                'source': 'test_source',
                'source_output': 'test',
                'sink_input': 'test',
                'sink': 'test_convertor'
            },
            {
                'source': 'test_convertor',
                'source_output': 'test',
                'sink_input': 'test',
                'sink': 'energy_demand'
            }
        ],
        'narratives': []
    })

    store.write_model(get_sector_model)

    store.write_initial_conditions('energy_demand', [])
    data = {
        'water_asset_a': {
            'name': 'water_asset_a',
            'build_year': 2010,
            'capacity': 50,
            'location': None,
            'sector': ''
        },
        'water_asset_b': {
            'name': 'water_asset_b',
            'build_year': 2015,
            'capacity': 150,
            'location': None,
            'sector': ''
        },
        'water_asset_c': {
            'name': 'water_asset_c',
            'capacity': 100,
            'build_year': 2015,
            'location': None,
            'sector': ''
        }
    }
    store.write_interventions('energy_demand', data)

    da = DataArray(parameter_spec, np.array(42))
    store.write_model_parameter_default('energy_demand', 'smart_meter_savings', da)
    return store


@fixture(scope='function')
def mock_sector_model():
    mock_sector_model = MagicMock()
    spec = Mock(spec=Spec)
    spec.name = Mock(return_value='test_output')
    type(spec).shape = PropertyMock(return_value=(2, 2))
    type(spec).name = PropertyMock(return_value='test_output')
    type(spec).dtype = PropertyMock(return_value='float')
    type(mock_sector_model).outputs = PropertyMock(return_value={'test_output': spec})
    type(mock_sector_model).name = PropertyMock(return_value='energy_demand')
    return mock_sector_model


@fixture(scope='function')
def mock_sos_model(mock_sector_model):
    mock_sos_model = MagicMock(outputs=[('energy_demand', 'test_output')])
    mock_sos_model.name = 'test_sos_model'
    mock_sos_model.models = [mock_sector_model]
    mock_sos_model.get_model = Mock(return_value=mock_sector_model)
    return mock_sos_model


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
    return EmptySectorModel('energy_demand')


@fixture(scope='function')
def mock_model():
    """Sector model with parameter, input, output, dependency
    """
    model = EmptySectorModel('energy_demand')
    model.add_parameter(
        Spec(
            name='smart_meter_savings',
            description='The savings from smart meters',
            abs_range=(0, 100),
            exp_range=(3, 10),
            unit='%',
            dtype='float'
        )
    )

    spec = Spec(
        name='test',
        dtype='float',
        dims=['lad', 'technology_type'],
        coords={'lad': [1, 2], 'technology_type': ['water_meter', 'electricity_meter']}
    )
    model.add_input(spec)
    model.add_output(spec)

    source_model = EmptySectorModel('test_source')
    source_model.add_output(spec)

    return model


@fixture(scope='function')
def mock_model_with_conversion():
    """Sector model with dependency via convertor
    """
    source = EmptySectorModel('test_source')
    convertor = EmptySectorModel('test_convertor')
    model = EmptySectorModel('energy_demand')

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

    return model


class TestDataHandle():
    """
    """

    def test_create(self, mock_model, mock_store):
        """should be created with a Store
        """
        DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

    def test_get_data(self, mock_store, mock_model):
        """should allow read access to input data
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        spec = mock_model.inputs['test']
        da = DataArray(spec, data)

        mock_store.write_results(
            da,
            1,
            'test_source',  # write source model results
            2015,
            None
        )
        actual = data_handle.get_data("test")

        np.testing.assert_equal(actual, da)

    def test_get_data_with_conversion(self, mock_store, mock_model_with_conversion):
        """should convert liters to milliliters (1 -> 0.001)
        """
        modelrun_name = 2
        data_handle = DataHandle(
            mock_store, modelrun_name, 2015, [2015, 2020], mock_model_with_conversion)
        data = np.array([[0.001, 0.003], [0.002, 0.004]])
        spec = mock_model_with_conversion.inputs['test']

        da = DataArray(spec, data)

        mock_store.write_results(
            da,
            modelrun_name,
            'test_convertor',  # write results as though from convertor
            2015,
            None
        )
        actual = data_handle.get_data("test")

        expected = DataArray(spec, data)

        np.testing.assert_equal(actual, expected)

    def test_get_base_timestep_data(self, mock_store, mock_model):
        """should allow read access to input data from base timestep
        """
        data_handle = DataHandle(mock_store, 1, 2025, [2015, 2020, 2025], mock_model)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        spec = mock_model.inputs['test']
        da = DataArray(spec, data)

        mock_store.write_results(
            da,
            1,
            'test_source',  # write source model results
            2015,  # base timetep
            None
        )

        actual = data_handle.get_base_timestep_data("test")
        expected = DataArray(spec, data)
        np.testing.assert_equal(actual, expected)

    def test_get_base_timestep_data_before_model_run(self, mock_store, mock_model):
        """Prior to a model run, there is no current timestep

        This should allow read access to input data from base timestep
        """
        data_handle = DataHandle(mock_store, 1, None, [2015, 2020, 2025], mock_model)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        spec = mock_model.inputs['test']
        da = DataArray(spec, data)

        mock_store.write_results(
            da,
            1,
            'test_source',  # write source model results
            2015,  # base timetep
            None
        )

        actual = data_handle.get_base_timestep_data("test")
        expected = DataArray(spec, data)
        np.testing.assert_equal(actual, expected)

    def test_get_previous_timestep_data(self, mock_store, mock_model):
        """should allow read access to input data from previous timestep
        """
        data_handle = DataHandle(mock_store, 1, 2025, [2015, 2020, 2025], mock_model)
        data = np.random.rand(*mock_model.inputs['test'].shape)
        spec = mock_model.inputs['test']
        da = DataArray(spec, data)

        mock_store.write_results(
            da,
            1,
            'test_source',  # write source model results
            2020,  # previous timetep
            None
        )
        actual = data_handle.get_previous_timestep_data("test")

        expected = DataArray(spec, data)

        assert actual == expected

    def test_get_data_with_square_brackets(self, mock_store, mock_model):
        """should allow dict-like read access to input data
        """
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        data = np.random.rand(*mock_model.inputs['test'].shape)

        spec = mock_model.inputs['test']
        da = DataArray(spec, data)

        mock_store.write_results(
            da,
            1,
            'test_source',  # write source model results
            2015,  # current timetep
            None
        )
        actual = data_handle["test"]
        assert actual == da

    def test_set_data(self, mock_store, mock_model):
        """should allow write access to output data
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        spec = mock_model.outputs['test']
        da = DataArray(spec, data)

        data_handle.set_results("test", data)
        actual = mock_store.read_results(
            1,
            'energy_demand',  # read results from model
            mock_model.outputs['test'],
            2015,
            None
        )
        np.testing.assert_equal(actual.as_ndarray(), data)
        assert actual == da

    def test_set_data_wrong_shape(self, mock_store, mock_model):
        """should allow write access to output data
        """
        data = np.array([[1.0, 1.0]])  # regions is 1, intervals is 1 not 2

        spec = mock_model.outputs['test']
        with raises(SmifDataMismatchError) as ex:
            DataArray(spec, data)

        msg = "Data shape (1, 2) does not match spec " \
              "(2, 2)"
        assert msg in str(ex)

    def test_set_data_with_square_brackets(self, mock_store, mock_model):
        """should allow dict-like write access to output data
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        spec = mock_model.outputs['test']
        da = DataArray(spec, data)

        data_handle["test"] = data
        actual = mock_store.read_results(
            1,
            'energy_demand',  # read results from model
            mock_model.outputs['test'],
            2015
        )
        np.testing.assert_equal(actual.as_ndarray(), data)
        assert actual == da

    def test_set_data_with_square_brackets_raises(self, mock_store, mock_model):
        """should allow dict-like write access to output data
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        spec = mock_model.outputs['test']
        da = DataArray(spec, data)

        with raises(TypeError) as err:
            data_handle["test"] = da

        assert "Pass in a numpy array" in str(err)


class TestDataHandleState():
    """Test handling of initial conditions, decision interventions and intervention state.
    """

    def test_get_state(self, mock_store, mock_model):
        """should get decision module state for given timestep/decision_iteration

        A call to ``get_state`` method on the data handle calls the read_state
        method of the store with arguments for model run name, current
        timestep and decision iteration.
        """
        mock_store.read_state = Mock(return_value=[{'name': 'test', 'build_year': 2010}])
        mock_store.write_interventions('energy_demand', [{
            'name': 'test',
            'capital_cost': {'value': 2500, 'unit': '£/GW'}
        }])
        data_handle = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        expected = [{
            'name': 'test',
            'build_year': 2010
        }]
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
        mock_store.write_state(state, 1, 2015, None)
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

        mock_store.write_state(state, 1, 2015, None)

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
        with raises(SmifTimestepResolutionError) as ex:
            data_handle.previous_timestep
        assert 'no previous timestep' in str(ex)


class TestDataHandleGetResults:

    def test_get_results_sectormodel(self, mock_store,
                                     mock_sector_model):
        """Get results from a sector model
        """
        store = mock_store
        spec = mock_sector_model.outputs['test_output']

        data = np.array([[1, 2.], [3., 4]])
        da = DataArray(spec, data)

        store.write_results(da, 1, 'energy_demand', 2010)

        dh = DataHandle(mock_store, 1, 2010, [2010], mock_sector_model)
        actual = dh.get_results('test_output')
        assert actual == DataArray(spec, data)

    def test_get_results_no_output_sector(self, mock_store,
                                          mock_sector_model):

        with raises(KeyError):
            dh = DataHandle(mock_store, 1, 2010, [2010], mock_sector_model)
            dh.get_results('no_such_output')


class TestDataHandleGetParameters:

    def test_load_parameter_defaults(self, mock_store, mock_model):

        dh = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        actual = dh.get_parameter('smart_meter_savings')
        spec = Spec.from_dict(
            {
                'name': 'smart_meter_savings',
                'description': "Difference in floor area per person"
                               "in end year compared to base year",
                'absolute_range': [0, float('inf')],
                'expected_range': [0.5, 2],
                'unit': '%',
                'dtype': 'float'
            })
        expected = DataArray(spec, np.array(42, dtype=float))

        assert actual == expected

    def test_load_parameters_override(self, mock_store, mock_model):

        mock_store.update_model_run(1, {
            'name': 1,
            'narratives': {'test_narrative': ['high_tech_dsm']},
            'sos_model': 'test_sos_model',
            'scenarios': {}})
        dh = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        actual = dh.get_parameter('smart_meter_savings')
        spec = Spec.from_dict(
            {
                'name': 'smart_meter_savings',
                'description': "Difference in floor area per person"
                               "in end year compared to base year",
                'absolute_range': [0, float('inf')],
                'expected_range': [0.5, 2],
                'unit': '%',
                'dtype': 'float'
            })
        expected = DataArray(spec, np.array(99))

        assert actual == expected

    def test_load_parameters_override_ordered(self, mock_store, mock_model):
        """Parameters in a narrative variants listed later override parameters
        contained in earlier variants
        """

        mock_store.update_model_run(1, {
            'name': 1,
            'narratives': {'test_narrative': ['first_variant',
                                              'second_variant']},
            'sos_model': 'test_sos_model',
            'scenarios': {}})

        sos_model = mock_store.read_sos_model('test_sos_model')
        sos_model['narratives'] = [{
            'name': 'test_narrative',
            'description': 'a narrative config',
            'sos_model': 'test_sos_model',
            'provides': {'energy_demand': ['smart_meter_savings']},
            'variants': [
                {
                    'name': 'first_variant',
                    'description': 'This variant should be overridden',
                    'data': {'smart_meter_savings': 'filename.csv'}},
                {
                    'name': 'second_variant',
                    'description': 'This variant should override the first',
                    'data': {'smart_meter_savings': 'filename.csv'}}
            ]
        }]
        mock_store.update_sos_model('test_sos_model', sos_model)

        parameter_spec = Spec(
            name='smart_meter_savings',
            dtype='float',
            unit='%'
        )
        first_variant = DataArray(parameter_spec, np.array(1))
        mock_store.write_narrative_variant_data(
            'test_sos_model', 'test_narrative', 'first_variant', first_variant)

        second_variant = DataArray(parameter_spec, np.array(2))
        mock_store.write_narrative_variant_data(
            'test_sos_model', 'test_narrative', 'second_variant', second_variant)

        dh = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)

        actual = dh.get_parameter('smart_meter_savings')

        assert actual == second_variant

    def test_load_parameters_partial_override(self, mock_store, mock_model):
        # add parameter to model
        param_spec = Spec(
            name='multi_savings',
            description='The savings from various technologies',
            abs_range=(0, 100),
            exp_range=(3, 10),
            dims=['technology_type'],
            coords={'technology_type': ['water_meter', 'electricity_meter']},
            unit='%',
            dtype='float'
        )
        mock_model.add_parameter(param_spec)
        mock_store.update_model(mock_model.name, mock_model.as_dict())

        # default values
        param_defaults = DataArray(param_spec, np.array([3, 3]))
        mock_store.write_model_parameter_default(
            mock_model.name, param_spec.name, param_defaults)

        # add narrative to sosmodel
        sos_model = mock_store.read_sos_model('test_sos_model')
        sos_model['narratives'] = [{
            'name': 'test_narrative',
            'description': 'a narrative config',
            'sos_model': 'test_sos_model',
            'provides': {'energy_demand': ['multi_savings']},
            'variants': [
                {
                    'name': 'low_multi_save',
                    'description': 'Low values',
                    'data': {'multi_savings': 'multi_savings.csv'}
                }
            ]
        }]
        mock_store.update_sos_model('test_sos_model', sos_model)

        # narrative values - one NaN i.e. not overridden
        param_narrative = DataArray(param_spec, np.array([np.nan, 99]))
        mock_store.write_narrative_variant_data(
            'test_sos_model', 'test_narrative', 'low_multi_save', param_narrative)

        # expect combination
        expected = DataArray(param_spec, np.array([3, 99]))

        mock_store.update_model_run(1, {
            'name': 1,
            'narratives': {'test_narrative': ['low_multi_save']},
            'sos_model': 'test_sos_model',
            'scenarios': {}
        })

        dh = DataHandle(mock_store, 1, 2015, [2015, 2020], mock_model)
        actual = dh.get_parameter('multi_savings')

        assert actual == expected


class TestResultsHandle:

    def test_get_results_sos_model(self, mock_store, mock_sector_model, mock_sos_model):
        """Get results from a sector model within a sos model
        """
        store = mock_store
        spec = mock_sector_model.outputs['test_output']

        da = DataArray(spec, np.array([[42, 42], [1, 2.]]))

        store.write_results(da, 'test_modelrun', 'energy_demand', 2010, None)

        dh = ResultsHandle(mock_store, 'test_modelrun', mock_sos_model, 2010)
        actual = dh.get_results('energy_demand', 'test_output', 2010, None)
        spec = mock_sector_model.outputs['test_output']
        assert actual == da

    def test_get_results_no_output_sos(self, mock_store, mock_sos_model):
        with raises(KeyError):
            dh = ResultsHandle(mock_store, 'test_modelrun', mock_sos_model, 2010)
            dh.get_results('energy_demand', 'no_such_output', 2010, None)

    def test_get_results_wrong_name_sos(self, mock_store, mock_sos_model):
        with raises(KeyError):
            dh = ResultsHandle(mock_store, 'test_modelrun', mock_sos_model, 2010)
            dh.get_results('no_such_model', 'test_output', 2010, None)

    def test_get_results_not_exists(self, mock_store, mock_sos_model, mock_model):
        store = mock_store

        spec = mock_model.outputs['test']
        da = DataArray(spec, np.array([[42, 42], [69, 69]]))

        store.write_results(da, 'test_modelrun', 'energy_demand', 2010, None)
        dh = ResultsHandle(store, 'test_modelrun', mock_sos_model, 2100)
        with raises(SmifDataError):
            dh.get_results('energy_demand', 'test_output', 2099, None)
