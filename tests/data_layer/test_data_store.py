"""Test all DataStore implementations
"""
from copy import deepcopy

import numpy as np
from pytest import fixture, mark, param, raises
from smif.data_layer.data_array import DataArray
from smif.data_layer.database_interface import DbDataStore
from smif.data_layer.file.file_data_store import CSVDataStore, ParquetDataStore
from smif.data_layer.memory_interface import MemoryDataStore
from smif.exception import SmifDataNotFoundError
from smif.metadata import Spec


@fixture(
    params=[
        'memory',
        'file_csv',
        'file_parquet',
        param('database', marks=mark.skip)]
    )
def handler(request, setup_empty_folder_structure):
    if request.param == 'memory':
        handler = MemoryDataStore()
    elif request.param == 'file_csv':
        base_folder = setup_empty_folder_structure
        handler = CSVDataStore(base_folder)
    elif request.param == 'file_parquet':
        base_folder = setup_empty_folder_structure
        handler = ParquetDataStore(base_folder)
    elif request.param == 'database':
        handler = DbDataStore()
        raise NotImplementedError

    return handler


class TestDataArray():
    """Read and write DataArray
    """
    def test_read_write_data_array(self, handler, scenario):
        spec_config = deepcopy(scenario['provides'][0])
        spec_config['dims'] = ['timestep'] + spec_config['dims']
        spec_config['coords']['timestep'] = [{'name': 2010}]
        spec = Spec.from_dict(spec_config)
        data = np.array([[0, 1]], dtype='float')
        da = DataArray(spec, data)
        handler.write_scenario_variant_data('mortality.csv', da)

        spec_config = deepcopy(scenario['provides'][0])
        spec = Spec.from_dict(spec_config)
        data = np.array([0, 1], dtype='float')
        expected = DataArray(spec, data)

        actual = handler.read_scenario_variant_data('mortality.csv', spec, 2010)
        assert actual == expected
        np.testing.assert_array_equal(actual.as_ndarray(), expected.as_ndarray())

    def test_read_write_data_array_all(self, handler, scenario):
        spec = Spec.from_dict(deepcopy(scenario['provides'][0]))

        spec_with_t = scenario['provides'][0]
        spec_with_t['dims'].insert(0, 'timestep')
        spec_with_t['coords']['timestep'] = [2010, 2015]
        spec_with_t = Spec.from_dict(spec_with_t)
        da = DataArray(spec_with_t, np.array([[0, 1], [2, 3]], dtype='float'))

        handler.write_scenario_variant_data('mortality.csv', da)
        actual = handler.read_scenario_variant_data('mortality.csv', spec_with_t)
        expected = np.array([[0, 1], [2, 3]], dtype='float')
        np.testing.assert_array_equal(actual.as_ndarray(), expected)

        da_2010 = handler.read_scenario_variant_data('mortality.csv', spec, 2010)
        expected = np.array([0, 1], dtype='float')
        np.testing.assert_array_equal(da_2010.as_ndarray(), expected)

        da_2015 = handler.read_scenario_variant_data('mortality.csv', spec, 2015)
        expected = np.array([2, 3], dtype='float')
        np.testing.assert_array_equal(da_2015.as_ndarray(), expected)

    def test_read_zero_d_from_timeseries(self, handler):
        """Read a single value
            timestep,param
            2010,0
            2015,1
        """
        # write data for multiple timesteps by adding 'timestep' dim to the spec
        data = np.array([0, 1], dtype=float)
        write_spec = Spec(
            name='param',
            dims=['timestep'],
            coords={'timestep': [2010, 2015]},
            dtype='float'
        )
        da = DataArray(write_spec, data)
        handler.write_scenario_variant_data('param', da)

        read_spec = Spec(
            name='param',
            dims=[],
            coords={},
            dtype='float'
        )
        actual = handler.read_scenario_variant_data('param', read_spec, 2010).data
        assert actual == np.array(0.0)
        actual = handler.read_scenario_variant_data('param', read_spec, 2015).data
        assert actual == np.array(1.0)

    def test_read_data_array_missing_timestep(self, handler, scenario):
        data = np.array([[0, 1]], dtype=float)
        spec_config = deepcopy(scenario['provides'][0])
        spec_config['dims'] = ['timestep'] + spec_config['dims']
        spec_config['coords']['timestep'] = [{'name': 2010}]
        spec = Spec.from_dict(spec_config)

        da = DataArray(spec, data)

        handler.write_scenario_variant_data('mortality.csv', da)
        msg = "not found for timestep 2011"
        with raises(SmifDataNotFoundError) as ex:
            handler.read_scenario_variant_data('mortality.csv', spec, 2011)
        assert msg in str(ex.value)

    def test_string_data(self, handler):
        spec = Spec(
            name='string_data',
            dims=['timestep', 'zones'],
            coords={'timestep': [2010], 'zones': ['a', 'b', 'c']},
            dtype='object'
        )
        data = np.array([['alpha', 'beta', 'γάμμα']], dtype='object')
        expected = DataArray(spec, data)

        handler.write_scenario_variant_data('key', expected)
        actual = handler.read_scenario_variant_data('key', spec)
        assert actual == expected


class TestInitialConditions():
    """Read and write initial conditions
    """
    def test_read_write_initial_conditions(self, handler, initial_conditions):

        expected = initial_conditions

        handler.write_initial_conditions('initial_conditions.csv', initial_conditions)
        actual = handler.read_initial_conditions(['initial_conditions.csv'])

        assert actual == expected


class TestInterventions():
    """Read and write interventions
    """
    def test_read_write_interventions(self, handler, interventions):

        expected = interventions
        handler.write_interventions('my_intervention.csv', interventions)
        actual = handler.read_interventions(['my_intervention.csv'])

        assert actual == expected


class TestState():
    """Read and write state
    """
    def test_read_write_state(self, handler, state):
        expected = state
        modelrun_name = 'test_modelrun'
        timestep = 2020
        decision_iteration = None

        handler.write_state(expected, modelrun_name, timestep, decision_iteration)
        actual = handler.read_state(modelrun_name, timestep, decision_iteration)
        assert actual == expected

    def test_read_write_empty_state(self, handler):
        expected = []
        modelrun_name = 'test_modelrun'
        timestep = 2020
        decision_iteration = None

        handler.write_state(expected, modelrun_name, timestep, decision_iteration)
        actual = handler.read_state(modelrun_name, timestep, decision_iteration)
        assert actual == expected


class TestCoefficients():
    """Read/write conversion coefficients
    """
    def test_read_write_coefficients(self, handler):

        with raises(SmifDataNotFoundError):
            handler.read_coefficients('from_dim_name', 'to_dim_name')

        expected = np.array([[2]])
        handler.write_coefficients('from_dim_name', 'to_dim_name', expected)

        actual = handler.read_coefficients('from_dim_name', 'to_dim_name')
        np.testing.assert_equal(actual, expected)


class TestResults():
    """Read/write results and prepare warm start
    """
    def test_read_write_results(self, handler, sample_results):
        output_spec = sample_results.spec
        modelrun_name = 'test_modelrun'
        model_name = 'energy'
        timestep = 2010

        handler.write_results(sample_results, modelrun_name, model_name, timestep)
        results_out = handler.read_results(modelrun_name, model_name, output_spec, timestep)

        assert results_out == sample_results

    def test_available_results(self, handler, sample_results):
        """Available results should return an empty list if none are available
        develop
        """
        assert handler.available_results('test_modelrun') == []
        handler.write_results(sample_results, 'test_modelrun', 'energy', 2010, 0)
        handler.write_results(sample_results, 'test_modelrun', 'energy', 2015, 0)
        handler.write_results(sample_results, 'test_modelrun', 'energy', 2015, 1)

        # keys should be (timestep, decision_iteration, model_name, output_name)
        assert sorted(handler.available_results('test_modelrun')) == [
            (2010, 0, 'energy', sample_results.spec.name),
            (2015, 0, 'energy', sample_results.spec.name),
            (2015, 1, 'energy', sample_results.spec.name)
        ]

        # delete one
        handler.delete_results('test_modelrun', 'energy', sample_results.spec.name, 2010, 0)
        assert sorted(handler.available_results('test_modelrun')) == [
            (2015, 0, 'energy', sample_results.spec.name),
            (2015, 1, 'energy', sample_results.spec.name)
        ]

        # clear all (no error if re-deleting)
        handler.delete_results('test_modelrun', 'energy', sample_results.spec.name, 2010, 0)
        handler.delete_results('test_modelrun', 'energy', sample_results.spec.name, 2015, 0)
        handler.delete_results('test_modelrun', 'energy', sample_results.spec.name, 2015, 1)
        assert not handler.available_results('test_modelrun')

    def test_read_results_raises(self, handler, sample_results):
        modelrun_name = 'test_modelrun'
        model_name = 'energy'
        timestep = 2010
        output_spec = sample_results.spec

        handler.write_results(sample_results, modelrun_name, model_name, timestep=timestep)

        with raises(SmifDataNotFoundError):
            handler.read_results(modelrun_name, model_name, output_spec, 2020)


class TestDataExists():
    """Check that model run data exists
    """
    def test_strategy_data_exists(self, handler, strategies):
        for strategy in strategies:
            if strategy['type'] == 'pre-specified-planning':
                new_strategy = {'filename': 'strategy'}
                assert not handler.strategy_data_exists(new_strategy)
                handler.write_strategy_interventions(new_strategy, strategy['interventions'])

                assert handler.strategy_data_exists(new_strategy)

    def test_scenario_variant_data_exists(self, handler, sample_scenario_data):
        assert not handler.scenario_variant_data_exists('scenario_variant_data')

        key = next(iter(sample_scenario_data))
        scenario_name, variant_name, variable = key
        scenario_variant_data = sample_scenario_data[key]
        handler.write_scenario_variant_data('scenario_variant_data', scenario_variant_data)

        assert handler.scenario_variant_data_exists('scenario_variant_data')

    def test_narrative_variant_data_exists(self, handler, sample_narrative_data):
        assert not handler.narrative_variant_data_exists('narrative_variant_data')
        # pick out single sample
        key = (
            'energy',
            'technology',
            'high_tech_dsm',
            'smart_meter_savings'
        )
        sos_model_name, narrative_name, variant_name, param_name = key
        narrative_variant_data = sample_narrative_data[key]
        handler.write_narrative_variant_data('narrative_variant_data', narrative_variant_data)

        assert handler.narrative_variant_data_exists('narrative_variant_data')

    def test_model_parameter_default_data_exists(self, handler, get_multidimensional_param):
        assert not handler.model_parameter_default_data_exists('parameter_default')
        param_data = get_multidimensional_param
        handler.write_model_parameter_default('parameter_default', param_data)

        assert handler.model_parameter_default_data_exists('parameter_default')

    def test_interventions_data_exists(self, handler, interventions):
        assert not handler.interventions_data_exists('interventions')
        handler.write_interventions('interventions', interventions)

        assert handler.interventions_data_exists('interventions')

    def test_initial_conditions_data_exists(self, handler, initial_conditions):
        assert not handler.initial_conditions_data_exists('initial_conditions')
        handler.write_initial_conditions('initial_conditions', initial_conditions)

        assert handler.initial_conditions_data_exists('initial_conditions')
