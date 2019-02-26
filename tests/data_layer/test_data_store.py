"""Test all DataStore implementations
"""
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
        data = np.array([0, 1], dtype=float)
        spec = Spec.from_dict(scenario['provides'][0])

        da = DataArray(spec, data)

        handler.write_scenario_variant_data('mortality.csv', da, 2010)
        actual = handler.read_scenario_variant_data('mortality.csv', spec, 2010)

        assert actual == da

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
        data = np.array([0, 1], dtype=float)
        spec = Spec.from_dict(scenario['provides'][0])

        da = DataArray(spec, data)

        handler.write_scenario_variant_data('mortality.csv', da, 2010)
        msg = "not found for timestep 2011"
        with raises(SmifDataNotFoundError) as ex:
            handler.read_scenario_variant_data('mortality.csv', spec, 2011)
        assert msg in str(ex)

    def test_string_data(self, handler):
        spec = Spec(
            name='string_data',
            dims=['zones'],
            coords={'zones': ['a', 'b', 'c']},
            dtype='object'
        )
        data = np.array(['alpha', 'beta', 'γάμμα'], dtype='object')
        expected = DataArray(spec, data)

        handler.write_scenario_variant_data('key', expected, 2010)
        actual = handler.read_scenario_variant_data('key', spec, 2010)
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


class TestCoefficients():
    """Read/write conversion coefficients
    """
    def test_read_write_coefficients(self, conversion_source_spec, conversion_sink_spec,
                                     handler):

        with raises(SmifDataNotFoundError):
            handler.read_coefficients(conversion_source_spec, conversion_sink_spec)

        expected = np.array([[2]])
        handler.write_coefficients(conversion_source_spec, conversion_sink_spec, expected)

        actual = handler.read_coefficients(conversion_source_spec, conversion_sink_spec)
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

        # keys should be (timestep, decision_iteration, model_name, output_name)
        assert handler.available_results('test_modelrun') == \
            [(2010, 0, 'energy', sample_results.spec.name)]

    def test_read_results_raises(self, handler, sample_results):
        modelrun_name = 'test_modelrun'
        model_name = 'energy'
        timestep = 2010
        output_spec = sample_results.spec

        handler.write_results(sample_results, modelrun_name, model_name, timestep=timestep)

        with raises(SmifDataNotFoundError):
            handler.read_results(modelrun_name, model_name, output_spec, 2020)
