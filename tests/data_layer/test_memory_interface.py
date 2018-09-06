from copy import copy

import numpy as np
from pytest import fixture, raises
from smif.data_layer import DataExistsError, MemoryInterface
from smif.metadata import Spec


class TestModelRuns:
    """Read, write, update model runs
    """
    @fixture(scope='function')
    def model_run(self):
        return {
            'name': 'unique_model_run_name'
        }

    @fixture(scope='function')
    def modelrun_handler(self, model_run):
        handler = MemoryInterface()
        handler.write_model_run(model_run)
        return handler

    def test_read_model_runs(self, modelrun_handler, model_run):
        actual = modelrun_handler.read_model_runs()
        expected = [model_run]
        assert actual == expected

    def test_read_model_run(self, modelrun_handler, model_run):
        assert modelrun_handler.read_model_run('unique_model_run_name') == model_run

    def test_write_model_run(self, modelrun_handler, model_run):
        new_model_run = {
            'name': 'new_model_run_name',
            'description': 'Model run 2'
        }
        modelrun_handler.write_model_run(new_model_run)
        actual = modelrun_handler.read_model_runs()
        expected = [model_run, new_model_run]
        assert (actual == expected) or (list(reversed(actual)) == expected)

    def test_update_model_run(self, modelrun_handler):
        updated_model_run = {
            'name': 'unique_model_run_name',
            'description': 'Model run'
        }
        modelrun_handler.update_model_run('unique_model_run_name', updated_model_run)
        assert modelrun_handler.read_model_runs() == [updated_model_run]

    def test_delete_model_run(self, modelrun_handler):
        modelrun_handler.delete_model_run('unique_model_run_name')
        assert modelrun_handler.read_model_runs() == []


class TestSosModel:
    """Read, write, update, delete SosModel config
    """
    @fixture(scope='function')
    def sos_model_handler(self, get_sos_model):
        handler = MemoryInterface()
        handler.write_sos_model(get_sos_model)
        return handler

    def test_read_sos_models(self, sos_model_handler, get_sos_model):
        handler = sos_model_handler
        actual = handler.read_sos_models()
        expected = [get_sos_model]
        assert actual == expected

    def test_read_sos_model(self, sos_model_handler, get_sos_model):
        assert sos_model_handler.read_sos_model('energy') == get_sos_model

    def test_write_sos_model(self, sos_model_handler, get_sos_model):
        new_sos_model = copy(get_sos_model)
        new_sos_model['name'] = 'another_sos_model'
        sos_model_handler.write_sos_model(new_sos_model)
        actual = sos_model_handler.read_sos_models()
        expected = [get_sos_model, new_sos_model]
        assert (actual == expected) or (list(reversed(actual)) == expected)

    def test_write_existing_sos_model(self, sos_model_handler):
        handler = sos_model_handler
        with raises(DataExistsError):
            handler.write_sos_model({'name': 'energy'})

    def test_update_sos_model(self, sos_model_handler, get_sos_model):
        updated_sos_model = copy(get_sos_model)
        updated_sos_model['sector_models'] = ['energy_demand']
        sos_model_handler.update_sos_model('energy', updated_sos_model)
        assert sos_model_handler.read_sos_models() == [updated_sos_model]

    def test_delete_sos_model(self, sos_model_handler):
        sos_model_handler.delete_sos_model('energy')
        assert sos_model_handler.read_sos_models() == []


class TestSectorModel():
    """Read/write/update/delete SectorModel config
    """
    @fixture(scope='function')
    def sector_model_handler(self, get_sector_model):
        handler = MemoryInterface()
        handler.write_sector_model(get_sector_model)
        return handler

    def test_read_sector_models(self, sector_model_handler, get_sector_model):
        actual = sector_model_handler.read_sector_models()
        expected = [get_sector_model]
        assert actual == expected

    def test_read_sector_model(self, sector_model_handler, get_sector_model):
        actual = sector_model_handler.read_sector_model(get_sector_model['name'])
        expected = get_sector_model
        assert actual == expected

    def test_write_sector_model(self, sector_model_handler, get_sector_model):
        new_sector_model = copy(get_sector_model)
        new_sector_model['name'] = 'another_energy_sector_model'
        sector_model_handler.write_sector_model(new_sector_model)
        actual = sector_model_handler.read_sector_models()
        expected = [get_sector_model, new_sector_model]
        assert (actual == expected) or (list(reversed(actual)) == expected)

    def test_update_sector_model(self, sector_model_handler, get_sector_model):
        name = get_sector_model['name']
        expected = copy(get_sector_model)
        expected['inputs'] = ['energy_use']
        sector_model_handler.update_sector_model(name, expected)
        actual = sector_model_handler.read_sector_model(name)
        assert actual == expected

    def test_delete_sector_model(self, sector_model_handler, get_sector_model):
        sector_model_handler.delete_sector_model(get_sector_model['name'])
        expected = []
        actual = sector_model_handler.read_sector_models()
        assert actual == expected


class TestStrategies():
    """Read strategies data
    """
    def test_read_strategies(self):
        handler = MemoryInterface()
        expected = ['default_strategy']
        handler._strategies = {'default': expected[0]}
        actual = handler.read_strategies()
        assert actual == expected


class TestState():
    """Read and write state
    """
    def test_read_write_state(self):
        handler = MemoryInterface()
        expected = 'example_state'
        modelrun_name = 'test_modelrun'
        timestep = 2020
        decision_iteration = None

        handler.write_state(expected, modelrun_name, timestep, decision_iteration)
        actual = handler.read_state(modelrun_name, timestep, decision_iteration)
        assert actual == expected


class TestUnits():
    """Read units definitions
    """
    def test_read_units(self):
        handler = MemoryInterface()
        unit_def = ('kg', 'kilograms')
        handler._units = {'kg': unit_def}
        expected = [unit_def]
        actual = handler.read_unit_definitions()
        assert actual == expected


class TestDimensions():
    """Read/write/update/delete dimensions
    """
    @fixture(scope='function')
    def dimension(self):
        return {'name': 'category', 'elements': [1, 2, 3]}

    @fixture(scope='function')
    def dimension_handler(self, dimension):
        handler = MemoryInterface()
        handler.write_dimension(dimension)
        return handler

    def test_read_dimensions(self, dimension_handler, dimension):
        assert dimension_handler.read_dimensions() == [dimension]

    def test_read_dimension(self, dimension_handler, dimension):
        assert dimension_handler.read_dimension('category') == dimension

    def test_write_dimension(self, dimension_handler, dimension):
        another_dimension = {'name': '3rd', 'elements': ['a', 'b']}
        dimension_handler.write_dimension(another_dimension)
        assert dimension_handler.read_dimensions() == [dimension, another_dimension]

    def test_update_dimension(self, dimension_handler, dimension):
        another_dimension = {'name': 'category', 'elements': [4, 5, 6]}
        dimension_handler.update_dimension('category', another_dimension)
        assert dimension_handler.read_dimensions() == [another_dimension]

    def test_delete_dimension(self, dimension_handler):
        dimension_handler.delete_dimension('category')
        assert dimension_handler.read_dimensions() == []


class TestCoefficients():
    """Read/write conversion coefficients
    """
    @fixture
    def source_spec(self):
        return Spec(name='a', dtype='float', unit='ml')

    @fixture
    def sink_spec(self):
        return Spec(name='b', dtype='float', unit='ml')

    @fixture(scope='function')
    def coefficients_handler(self, source_spec, sink_spec):
        handler = MemoryInterface()
        handler.write_coefficients(source_spec, sink_spec, np.array([[1]]))
        return handler

    def test_read_coefficients(self, source_spec, sink_spec, coefficients_handler):
        actual = coefficients_handler.read_coefficients(source_spec, sink_spec)
        expected = np.array([[1]])
        np.testing.assert_equal(actual, expected)

    def test_write_coefficients(self, source_spec, sink_spec, coefficients_handler):
        expected = np.array([[2]])
        coefficients_handler.write_coefficients(source_spec, sink_spec, expected)
        actual = coefficients_handler.read_coefficients(source_spec, sink_spec)
        np.testing.assert_equal(actual, expected)


class TestScenarios():
    """Read and write scenario data
    """
    @fixture
    def scenario(self):
        return {
            'name': 'mortality',
            'description': 'The annual mortality rate in UK population',
            'variants': {
                'low': {
                    'name': 'low',
                    'description': 'Mortality (Low)',
                    'data': {
                        'mortality': 'mortality_low.csv',
                    },
                }
            }
        }

    @fixture
    def scenario_handler(self, scenario):
        handler = MemoryInterface()
        handler.write_scenario(scenario)
        return handler

    def test_read_scenarios(self, scenario, scenario_handler):
        assert scenario_handler.read_scenarios() == [scenario]

    def test_read_scenario(self, scenario, scenario_handler):
        assert scenario_handler.read_scenario('mortality') == scenario

    def test_write_scenario(self, scenario, scenario_handler):
        another_scenario = {
            'name': 'fertility',
            'description': 'Projected annual fertility rates',
            'variants': {}
        }
        scenario_handler.write_scenario(another_scenario)
        assert scenario_handler.read_scenarios() == [scenario, another_scenario]

    def test_update_scenario(self, scenario, scenario_handler):
        another_scenario = {
            'name': 'mortality',
            'description': 'Projected annual mortality rates',
            'variants': {}
        }
        scenario_handler.update_scenario('mortality', another_scenario)
        assert scenario_handler.read_scenarios() == [another_scenario]

    def test_delete_scenario(self, scenario_handler):
        scenario_handler.delete_scenario('mortality')
        assert scenario_handler.read_scenarios() == []

    def test_read_scenario_variants(self, scenario_handler, scenario):
        actual = scenario_handler.read_scenario_variants('mortality')
        expected = [scenario['variants']['low']]
        assert actual == expected

    def test_read_scenario_variant(self, scenario_handler, scenario):
        actual = scenario_handler.read_scenario_variant('mortality', 'low')
        expected = scenario['variants']['low']
        assert actual == expected

    def test_write_scenario_variant(self, scenario_handler, scenario):
        new_variant = {
            'name': 'high',
            'description': 'Mortality (High)',
            'data': {
                'mortality': 'mortality_high.csv'
            }
        }
        scenario_handler.write_scenario_variant('mortality', new_variant)
        actual = scenario_handler.read_scenario_variants('mortality')
        expected = [new_variant, scenario['variants']['low']]
        assert (actual == expected) or (actual == list(reversed(expected)))

    def test_update_scenario_variant(self, scenario_handler, scenario):
        new_variant = {
            'name': 'low',
            'description': 'Mortality (Low)',
            'data': {
                'mortality': 'mortality_low.csv'
            }
        }
        scenario_handler.update_scenario_variant('mortality', 'low', new_variant)
        actual = scenario_handler.read_scenario_variants('mortality')
        expected = [new_variant]
        assert actual == expected

    def test_delete_scenario_variant(self, scenario_handler, scenario):
        scenario_handler.delete_scenario_variant('mortality', 'low')
        assert scenario_handler.read_scenario_variants('mortality') == []

    def test_read_scenario_variant_data(self, get_remapped_scenario_data):
        """Read from in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler._scenario_data[('test_scenario', 'variant', 'parameter', 2010)] = data
        assert handler.read_scenario_variant_data(
            'test_scenario', 'variant', 'parameter', 2010) == data

    def test_write_scenario_variant_data(self, get_remapped_scenario_data):
        """Write to in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler.write_scenario_variant_data(
            data, 'test_scenario', 'variant', 'parameter', 2010)
        assert handler._scenario_data[('test_scenario', 'variant', 'parameter', 2010)] == data


class TestNarratives():
    """Read and write narrative data
    """
    @fixture
    def narrative(self):
        return {
            'name': 'technology',
            'description': 'Describes the evolution of technology',
            'variants': {
                'high_tech_dsm': {
                    'name': 'high_tech_dsm',
                    'description': 'High takeup of smart technology on the demand side',
                    'data': {
                        'smart_meter_savings': 'high_tech_dsm.csv',
                    },
                }
            }
        }

    @fixture
    def narrative_handler(self, narrative):
        handler = MemoryInterface()
        handler.write_narrative(narrative)
        return handler

    def test_read_narratives(self, narrative, narrative_handler):
        assert narrative_handler.read_narratives() == [narrative]

    def test_read_narrative(self, narrative, narrative_handler):
        assert narrative_handler.read_narrative('technology') == narrative

    def test_write_narrative(self, narrative, narrative_handler):
        another_narrative = {
            'name': 'policy',
            'description': 'Parameters decribing policy effects on demand',
            'variants': {}
        }
        narrative_handler.write_narrative(another_narrative)
        assert narrative_handler.read_narratives() == [narrative, another_narrative]

    def test_update_narrative(self, narrative, narrative_handler):
        another_narrative = {
            'name': 'technology',
            'description': 'Technology development, adoption and diffusion',
            'variants': {}
        }
        narrative_handler.update_narrative('technology', another_narrative)
        assert narrative_handler.read_narratives() == [another_narrative]

    def test_delete_narrative(self, narrative_handler):
        narrative_handler.delete_narrative('technology')
        assert narrative_handler.read_narratives() == []

    def test_read_narrative_variants(self, narrative_handler, narrative):
        actual = narrative_handler.read_narrative_variants('technology')
        expected = [narrative['variants']['high_tech_dsm']]
        assert actual == expected

    def test_read_narrative_variant(self, narrative_handler, narrative):
        actual = narrative_handler.read_narrative_variant('technology', 'high_tech_dsm')
        expected = narrative['variants']['high_tech_dsm']
        assert actual == expected

    def test_write_narrative_variant(self, narrative_handler, narrative):
        new_variant = {
            'name': 'precautionary',
            'description': 'Slower take-up of smart demand-response technologies',
            'data': {
                'technology': 'precautionary.csv'
            }
        }
        narrative_handler.write_narrative_variant('technology', new_variant)
        actual = narrative_handler.read_narrative_variants('technology')
        expected = [new_variant, narrative['variants']['high_tech_dsm']]
        assert (actual == expected) or (actual == list(reversed(expected)))

    def test_update_narrative_variant(self, narrative_handler, narrative):
        new_variant = {
            'name': 'high_tech_dsm',
            'description': 'High takeup of smart technology on the demand side (v2)',
            'data': {
                'technology': 'high_tech_dsm_v2.csv'
            }
        }
        narrative_handler.update_narrative_variant('technology', 'high_tech_dsm', new_variant)
        actual = narrative_handler.read_narrative_variants('technology')
        expected = [new_variant]
        assert actual == expected

    def test_delete_narrative_variant(self, narrative_handler, narrative):
        narrative_handler.delete_narrative_variant('technology', 'high_tech_dsm')
        assert narrative_handler.read_narrative_variants('technology') == []

    def test_read_narrative_variant_data(self, get_remapped_scenario_data):
        """Read from in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler._narrative_data[('technology', 'high_tech_dsm', 'param', 2010)] = data
        assert handler.read_narrative_variant_data(
            'technology', 'high_tech_dsm', 'param', 2010) == data

    def test_write_narrative_variant_data(self, get_remapped_scenario_data):
        """Write to in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler.write_narrative_variant_data(
            data, 'technology', 'high_tech_dsm', 'param', 2010)
        assert handler._narrative_data[('technology', 'high_tech_dsm', 'param', 2010)] == data


class TestResults():
    """Read/write results and prepare warm start
    """
    @fixture(scope='function')
    def results_handler(self):
        handler = MemoryInterface()
        handler.write_model_run({
            'name': 'test_modelrun',
            'timesteps': [2010, 2015, 2020]
        })
        return handler

    def test_read_write_results(self, results_handler):
        results_in = np.array(1)
        modelrun_name = 'test_modelrun'
        model_name = 'energy'
        output_spec = Spec(name='energy_use', dtype='float')

        results_handler.write_results(results_in, modelrun_name, model_name, output_spec)
        results_out = results_handler.read_results(modelrun_name, model_name, output_spec)
        assert results_in == results_out

    def test_warm_start(self, results_handler):
        """Warm start should return initial timestep (actual warm start not implemented in
        MemoryInterface)
        """
        start = results_handler.prepare_warm_start('test_modelrun')
        assert start == 2010
