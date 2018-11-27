"""Test all DataStore implementations
"""
import numpy as np
from pytest import fixture, mark, param, raises
from smif.data_layer.data_array import DataArray
from smif.data_layer.database_interface import DbDataStore
from smif.data_layer.datafile_interface import CSVDataStore
from smif.data_layer.memory_interface import MemoryDataStore
from smif.exception import SmifDataNotFoundError
from smif.metadata import Spec


@fixture(
    params=[
        'memory',
        param('file', marks=mark.skip),
        param('database', marks=mark.skip)]
    )
def init_handler(request, setup_empty_folder_structure):
    if request.param == 'memory':
        handler = MemoryDataStore()
    elif request.param == 'file':
        base_folder = setup_empty_folder_structure
        handler = CSVDataStore(base_folder)
    elif request.param == 'database':
        handler = DbDataStore()
        raise NotImplementedError

    return handler


@fixture
def handler(init_handler, sample_narrative_data, get_sector_model,
            get_sector_model_parameter_defaults, conversion_source_spec, conversion_sink_spec,
            conversion_coefficients):
    handler = init_handler

    # parameter defaults
    for parameter_name, data in get_sector_model_parameter_defaults.items():
        handler.write_model_parameter_default(
            get_sector_model['name'], parameter_name, data)

    # narrative data
    for key, narrative_variant_data in sample_narrative_data.items():
        sos_model_name, narrative_name, variant_name, _ = key  # skip param name
        handler.write_narrative_variant_data(
            sos_model_name, narrative_name, variant_name, narrative_variant_data)

    # conversion coefficients
    handler.write_coefficients(
        conversion_source_spec, conversion_sink_spec, conversion_coefficients)
    return handler


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
    def test_read_coefficients(self, conversion_source_spec, conversion_sink_spec, handler,
                               conversion_coefficients):
        actual = handler.read_coefficients(conversion_source_spec, conversion_sink_spec)
        expected = conversion_coefficients
        np.testing.assert_equal(actual, expected)

    def test_write_coefficients(self, conversion_source_spec, conversion_sink_spec, handler):
        expected = np.array([[2]])
        handler.write_coefficients(conversion_source_spec, conversion_sink_spec, expected)
        actual = handler.read_coefficients(conversion_source_spec, conversion_sink_spec)
        np.testing.assert_equal(actual, expected)


class TestScenarios():
    """Read and write scenario data
    """
    def test_write_scenario_variant_data(self, handler, scenario):
        """Write to in-memory data
        """
        data = np.array([0, 1], dtype=float)

        spec = Spec.from_dict(scenario['provides'][0])
        da = DataArray(spec, data)

        handler.write_scenario_variant_data('mortality', 'low', da, 2010)

        actual = handler.read_scenario_variant_data('mortality', 'low',
                                                    'mortality', 2010)
        assert actual == da


class TestNarratives():
    """Read and write narrative data
    """
    def test_read_narrative_variant_data(self, handler, sample_narrative_data):
        """Read from in-memory data
        """
        actual = handler.read_narrative_variant_data('energy',
                                                     'technology',
                                                     'high_tech_dsm',
                                                     'smart_meter_savings')
        key = ('energy', 'technology', 'high_tech_dsm', 'smart_meter_savings')
        expected = sample_narrative_data[key]
        assert actual == expected

    def test_read_narrative_variant_data_raises_param(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_narrative_variant_data('energy',
                                                'technology',
                                                'high_tech_dsm',
                                                'not_a_parameter')

    def test_read_narrative_variant_data_raises_variant(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_narrative_variant_data('energy',
                                                'technology',
                                                'not_a_variant',
                                                'not_a_parameter')

    def test_read_narrative_variant_data_raises_narrative(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_narrative_variant_data('energy',
                                                'not_a_narrative',
                                                'not_a_variant',
                                                'not_a_parameter')

    def test_write_narrative_variant_data(self, handler, sample_narrative_data):
        """Write narrative variant data to file or memory
        """
        key = ('energy', 'technology', 'high_tech_dsm', 'smart_meter_savings')
        da = sample_narrative_data[key]
        handler.write_narrative_variant_data(
            'energy', 'technology', 'high_tech_dsm', da)

        actual = handler.read_narrative_variant_data(
            'energy', 'technology', 'high_tech_dsm', 'smart_meter_savings')

        assert actual == da

    def test_write_narrative_variant_data_timestep(self, handler, sample_narrative_data):
        """Write narrative variant data to file or memory
        """
        key = ('energy', 'technology', 'high_tech_dsm', 'smart_meter_savings')
        da = sample_narrative_data[key]
        handler.write_narrative_variant_data(
            'energy', 'technology', 'high_tech_dsm', da,
            timestep=2010)

        actual = handler.read_narrative_variant_data(
            'energy', 'technology', 'high_tech_dsm', 'smart_meter_savings', timestep=2010)

        assert actual == da


class TestResults():
    """Read/write results and prepare warm start
    """
    def test_read_write_results(self, handler):
        results_in = np.array(1, dtype=float)
        modelrun_name = 'test_modelrun'
        model_name = 'energy'
        timestep = 2010
        output_spec = Spec(name='energy_use', dtype='float')

        da = DataArray(output_spec, results_in)

        handler.write_results(da, modelrun_name, model_name, timestep)
        results_out = handler.read_results(modelrun_name, model_name, output_spec, timestep)

        expected = DataArray(output_spec, results_in)

        assert results_out == expected

    def test_warm_start(self, handler):
        """Warm start should return None if no results are available
        """
        start = handler.prepare_warm_start('test_modelrun')
        assert start is None

    def test_read_results_raises(self, handler):
        results_in = np.array(1)
        modelrun_name = 'test_modelrun'
        model_name = 'energy'
        timestep = 2010
        output_spec = Spec(name='energy_use', dtype='float')

        da = DataArray(output_spec, results_in)

        handler.write_results(da, modelrun_name, model_name, timestep=timestep)

        with raises(SmifDataNotFoundError):
            handler.read_results(modelrun_name, model_name, output_spec, 2020)
