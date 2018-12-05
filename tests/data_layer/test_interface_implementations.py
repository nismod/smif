from copy import copy

import numpy as np
from pytest import fixture, mark, param, raises
from smif.data_layer import (DatabaseInterface, DatafileInterface,
                             MemoryInterface)
from smif.data_layer.data_array import DataArray
from smif.exception import SmifDataExistsError, SmifDataNotFoundError
from smif.metadata import Spec


@fixture(
    params=[
        'memory',
        'file',
        param('database', marks=mark.skip)]
    )
def init_handler(request, setup_empty_folder_structure):
    if request.param == 'memory':
        handler = MemoryInterface()
    elif request.param == 'file':
        base_folder = setup_empty_folder_structure
        handler = DatafileInterface(base_folder, 'local_csv', validation=False)
    elif request.param == 'database':
        handler = DatabaseInterface()
        raise NotImplementedError

    return handler


@fixture
def handler(init_handler, minimal_model_run, get_sos_model, sample_narrative_data,
            get_sector_model, get_sector_model_parameter_defaults, strategies,
            unit_definitions, dimension, sample_dimensions, conversion_source_spec,
            conversion_sink_spec, conversion_coefficients, scenario):
    handler = init_handler

    # metadata
    handler.write_unit_definitions(unit_definitions)
    handler.write_dimension(dimension)
    for dim in sample_dimensions:
        handler.write_dimension(dim)

    # scenarios
    handler.write_scenario(scenario)

    # models
    handler.write_sector_model(get_sector_model)
    for parameter_name, data in get_sector_model_parameter_defaults.items():
        handler.write_sector_model_parameter_default(
            get_sector_model['name'], parameter_name, data)

    # sos models
    handler.write_sos_model(get_sos_model)
    for key, narrative_variant_data in sample_narrative_data.items():
        sos_model_name, narrative_name, variant_name, _ = key  # skip param name
        handler.write_narrative_variant_data(
            sos_model_name, narrative_name, variant_name, narrative_variant_data)

    # model runs
    handler.write_model_run(minimal_model_run)

    # planning
    handler.write_strategies('test_modelrun', strategies)

    # conversion coefficients
    handler.write_coefficients(
        conversion_source_spec, conversion_sink_spec, conversion_coefficients)
    return handler


class TestModelRuns:
    """Read, write, update model runs
    """
    def test_read_model_runs(self, handler, minimal_model_run):
        actual = handler.read_model_runs()
        expected = [minimal_model_run]
        assert actual == expected

    def test_read_model_run(self, handler, minimal_model_run):
        assert handler.read_model_run('test_modelrun') == minimal_model_run

    def test_read_model_run_sorted(self, handler, minimal_model_run):
        y_model_run = {'name': 'y'}
        z_model_run = {'name': 'z'}

        handler.write_model_run(z_model_run)
        handler.write_model_run(y_model_run)

        expected = [minimal_model_run, y_model_run, z_model_run]
        assert handler.read_model_runs() == expected

    def test_write_model_run(self, handler, minimal_model_run):
        new_model_run = {
            'name': 'new_model_run_name',
            'description': 'Model run 2'
        }
        handler.write_model_run(new_model_run)
        actual = handler.read_model_runs()
        expected = [minimal_model_run, new_model_run]
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_update_model_run(self, handler):
        updated_model_run = {
            'name': 'test_modelrun',
            'description': 'Model run'
        }
        handler.update_model_run('test_modelrun', updated_model_run)
        assert handler.read_model_runs() == [updated_model_run]

    def test_delete_model_run(self, handler):
        handler.delete_model_run('test_modelrun')
        assert handler.read_model_runs() == []


class TestSosModel:
    """Read, write, update, delete SosModel config
    """
    def test_read_sos_models(self, handler, get_sos_model):
        handler = handler
        actual = handler.read_sos_models()
        expected = [get_sos_model]
        assert actual == expected

    def test_read_sos_model(self, handler, get_sos_model):
        assert handler.read_sos_model('energy') == get_sos_model

    def test_write_sos_model(self, handler, get_sos_model):
        new_sos_model = copy(get_sos_model)
        new_sos_model['name'] = 'another_sos_model'
        handler.write_sos_model(new_sos_model)
        actual = handler.read_sos_models()
        expected = [get_sos_model, new_sos_model]
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_write_existing_sos_model(self, handler):
        handler = handler
        with raises(SmifDataExistsError):
            handler.write_sos_model({'name': 'energy'})

    def test_update_sos_model(self, handler, get_sos_model):
        updated_sos_model = copy(get_sos_model)
        updated_sos_model['sector_models'] = ['energy_demand']
        handler.update_sos_model('energy', updated_sos_model)
        assert handler.read_sos_models() == [updated_sos_model]

    def test_delete_sos_model(self, handler):
        handler.delete_sos_model('energy')
        assert handler.read_sos_models() == []


class TestSectorModel():
    """Read/write/update/delete SectorModel config
    """
    def test_read_sector_models(self, handler, get_sector_model):
        actual = handler.read_sector_models()
        expected = [get_sector_model]
        assert actual == expected

    def test_read_sector_models_no_coords(self, handler, get_sector_model,
                                          get_sector_model_no_coords):
        actual = handler.read_sector_models(skip_coords=True)
        expected = [get_sector_model_no_coords]
        assert actual == expected

    def test_read_sector_model(self, handler, get_sector_model):
        actual = handler.read_sector_model(get_sector_model['name'])
        expected = get_sector_model
        assert actual == expected

    def test_read_sector_model_no_coords(self, handler, get_sector_model,
                                         get_sector_model_no_coords):
        actual = handler.read_sector_model(get_sector_model['name'], skip_coords=True)
        expected = get_sector_model_no_coords
        assert actual == expected

    def test_sector_model_parameter_default(self, handler,
                                            get_sector_model_parameter_defaults):
        sector_model_name = 'energy_demand'
        parameter_name = 'smart_meter_savings'
        data = get_sector_model_parameter_defaults[parameter_name]
        handler.write_sector_model_parameter_default(sector_model_name, parameter_name, data)
        actual = handler.read_sector_model_parameter_default(sector_model_name, parameter_name)
        assert actual == data

    def test_write_sector_model(self, handler, get_sector_model):
        new_sector_model = copy(get_sector_model)
        new_sector_model['name'] = 'another_energy_sector_model'
        handler.write_sector_model(new_sector_model)
        actual = handler.read_sector_models()
        expected = [get_sector_model, new_sector_model]
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_update_sector_model(self, handler, get_sector_model):
        name = get_sector_model['name']
        expected = copy(get_sector_model)
        expected['description'] = ['Updated description']
        handler.update_sector_model(name, expected)
        actual = handler.read_sector_model(name)
        assert actual == expected

    def test_delete_sector_model(self, handler, get_sector_model):
        handler.delete_sector_model(get_sector_model['name'])
        expected = []
        actual = handler.read_sector_models()
        assert actual == expected


class TestStrategies():
    """Read strategies data
    """
    def test_read_strategies(self, handler, strategies):
        expected = strategies
        actual = handler.read_strategies('test_modelrun')
        assert sorted(actual, key=lambda d: d['description']) == \
            sorted(expected, key=lambda d: d['description'])


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


class TestUnits():
    """Read units definitions
    """
    def test_read_units(self, handler, unit_definitions):
        expected = unit_definitions
        actual = handler.read_unit_definitions()
        assert actual == expected


class TestDimensions():
    """Read/write/update/delete dimensions
    """
    def test_read_dimensions(self, handler, dimension, sample_dimensions):
        actual = handler.read_dimensions()
        expected = [dimension] + sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_read_dimension(self, handler, dimension):
        assert handler.read_dimension('category') == dimension

    def test_write_dimension(self, handler, dimension, sample_dimensions):
        another_dimension = {'name': '3rd', 'elements': ['a', 'b']}
        handler.write_dimension(another_dimension)
        actual = handler.read_dimensions()
        expected = [dimension, another_dimension] + sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_update_dimension(self, handler, dimension, sample_dimensions):
        another_dimension = {'name': 'category', 'elements': [4, 5, 6]}
        handler.update_dimension('category', another_dimension)
        actual = handler.read_dimensions()
        expected = [another_dimension] + sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_delete_dimension(self, handler, sample_dimensions):
        handler.delete_dimension('category')
        actual = handler.read_dimensions()
        expected = sample_dimensions
        assert sorted_by_name(actual) == sorted_by_name(expected)


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
    def test_read_scenarios(self, scenario, handler):
        actual = handler.read_scenarios()
        assert actual == [scenario]

    def test_read_scenarios_no_coords(self, scenario_no_coords, handler):
        assert handler.read_scenarios(skip_coords=True) == [scenario_no_coords]

    def test_read_scenario(self, scenario, handler):
        actual = handler.read_scenario('mortality')
        assert actual == scenario

    def test_read_scenario_no_coords(self, scenario_no_coords, handler):
        assert handler.read_scenario('mortality', skip_coords=True) == scenario_no_coords

    def test_write_scenario(self, scenario, handler):
        another_scenario = {
            'name': 'fertility',
            'description': 'Projected annual fertility rates',
            'variants': [],
            'provides': []
        }
        handler.write_scenario(another_scenario)
        actual = handler.read_scenario('fertility')
        expected = another_scenario
        assert actual == expected

    def test_update_scenario(self, scenario, handler):
        another_scenario = {
            'name': 'mortality',
            'description': 'Projected annual mortality rates',
            'variants': [],
            'provides': []
        }
        handler.update_scenario('mortality', another_scenario)
        assert handler.read_scenarios() == [another_scenario]

    def test_delete_scenario(self, handler):
        handler.delete_scenario('mortality')
        assert handler.read_scenarios() == []

    def test_read_scenario_variants(self, handler, scenario):
        actual = handler.read_scenario_variants('mortality')
        expected = scenario['variants']
        assert actual == expected

    def test_read_scenario_variant(self, handler, scenario):
        actual = handler.read_scenario_variant('mortality', 'low')
        expected = scenario['variants'][0]
        assert actual == expected

    def test_write_scenario_variant(self, handler, scenario):
        new_variant = {
            'name': 'high',
            'description': 'Mortality (High)',
            'data': {
                'mortality': 'mortality_high.csv'
            }
        }
        handler.write_scenario_variant('mortality', new_variant)
        actual = handler.read_scenario_variants('mortality')
        expected = [new_variant] + scenario['variants']
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_update_scenario_variant(self, handler, scenario):
        new_variant = {
            'name': 'low',
            'description': 'Mortality (Low)',
            'data': {
                'mortality': 'mortality_low.csv'
            }
        }
        handler.update_scenario_variant('mortality', 'low', new_variant)
        actual = handler.read_scenario_variants('mortality')
        expected = [new_variant]
        assert actual == expected

    def test_delete_scenario_variant(self, handler, scenario):
        handler.delete_scenario_variant('mortality', 'low')
        assert handler.read_scenario_variants('mortality') == []

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


def sorted_by_name(list_):
    """Helper to sort lists-of-dicts
    """
    return sorted(list_, key=lambda d: d['name'])
