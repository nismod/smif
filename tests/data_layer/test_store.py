"""Test the Store interface

Many methods simply proxy to config/metadata/data store implementations, but there is some
cross-coordination and there are some convenience methods implemented at this layer.
"""
import numpy.testing
from pytest import fixture
from smif.data_layer import Store
from smif.data_layer.memory_interface import (MemoryConfigStore,
                                              MemoryDataStore,
                                              MemoryMetadataStore)


@fixture
def store():
    """Store fixture
    """
    # implement each part using the memory classes, simpler than mocking
    # each other implementation of a part is tested fully by e.g. test_config_store.py
    return Store(
        config_store=MemoryConfigStore(),
        metadata_store=MemoryMetadataStore(),
        data_store=MemoryDataStore()
    )


class TestStoreConfig():
    def test_model_runs(self, store, minimal_model_run):
        # write
        store.write_model_run(minimal_model_run)
        # read all
        assert store.read_model_runs() == [minimal_model_run]
        # read one
        assert store.read_model_run(minimal_model_run['name']) == minimal_model_run
        # update
        store.update_model_run(minimal_model_run['name'], minimal_model_run)
        # delete
        store.delete_model_run(minimal_model_run['name'])
        assert store.read_model_runs() == []

    def test_read_model_run_sorted(self, store, minimal_model_run):
        y_model_run = {'name': 'y'}
        z_model_run = {'name': 'z'}

        store.write_model_run(minimal_model_run)
        store.write_model_run(z_model_run)
        store.write_model_run(y_model_run)

        expected = [minimal_model_run, y_model_run, z_model_run]
        assert store.read_model_runs() == expected

    def test_sos_models(self, store, get_sos_model, get_sector_model,
                        energy_supply_sector_model, sample_scenarios):
        # write
        store.write_model(get_sector_model)
        store.write_model(energy_supply_sector_model)
        for scenario in sample_scenarios:
            store.write_scenario(scenario)
        store.write_sos_model(get_sos_model)
        # read all
        assert store.read_sos_models() == [get_sos_model]
        # read one
        assert store.read_sos_model(get_sos_model['name']) == get_sos_model
        # update
        store.update_sos_model(get_sos_model['name'], get_sos_model)
        # delete
        store.delete_sos_model(get_sos_model['name'])
        assert store.read_sos_models() == []

    def test_models(self, store, get_sector_model, sample_dimensions):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        # write
        store.write_model(get_sector_model)
        # read all
        assert store.read_models() == [get_sector_model]
        # read one
        assert store.read_model(get_sector_model['name']) == get_sector_model
        # update
        store.update_model(get_sector_model['name'], get_sector_model)
        # delete
        store.delete_model(get_sector_model['name'])
        assert store.read_models() == []
        # teardown
        for dim in sample_dimensions:
            store.delete_dimension(dim['name'])

    def test_models_skip_coords(self, store, get_sector_model, get_sector_model_no_coords):
        # write
        store.write_model(get_sector_model)
        # read all
        assert store.read_models(skip_coords=True) == [get_sector_model_no_coords]
        # read one
        assert store.read_model(get_sector_model['name'], skip_coords=True) == \
            get_sector_model_no_coords
        # update
        store.update_model(get_sector_model['name'], get_sector_model)
        # delete
        store.delete_model(get_sector_model['name'])
        assert store.read_models() == []

    def test_scenarios(self, store, scenario, sample_dimensions):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        # write
        store.write_scenario(scenario)
        # read all
        assert store.read_scenarios() == [scenario]
        # read one
        assert store.read_scenario(scenario['name']) == scenario
        # update
        store.update_scenario(scenario['name'], scenario)
        # delete
        store.delete_scenario(scenario['name'])
        assert store.read_scenarios() == []
        # teardown
        for dim in sample_dimensions:
            store.delete_dimension(dim['name'])

    def test_scenarios_skip_coords(self, store, scenario, scenario_no_coords):
        # write
        store.write_scenario(scenario)
        # read all
        assert store.read_scenarios(skip_coords=True) == [scenario_no_coords]
        # read one
        assert store.read_scenario(scenario['name'], skip_coords=True) == \
            scenario_no_coords
        # update
        store.update_scenario(scenario['name'], scenario)
        # delete
        store.delete_scenario(scenario['name'])
        assert store.read_scenarios() == []

    def test_scenario_variants(self, store, scenario):
        scenario_name = scenario['name']
        old_variant = scenario['variants'][0]
        # write
        store.write_scenario(scenario)
        new_variant = {
            'name': 'high',
            'description': 'Mortality (High)',
            'data': {
                'mortality': 'mortality_high.csv'
            }
        }
        store.write_scenario_variant(scenario_name, new_variant)
        # read all
        both = store.read_scenario_variants(scenario_name)
        assert both == [old_variant, new_variant] or both == [new_variant, old_variant]
        # read one
        assert store.read_scenario_variant(scenario_name, old_variant['name']) == old_variant
        # update
        store.update_scenario_variant(scenario_name, old_variant['name'], old_variant)
        # delete
        store.delete_scenario_variant(scenario_name, old_variant['name'])
        assert store.read_scenario_variants(scenario_name) == [new_variant]

    def test_narratives(self, store, get_sos_model):
        store.write_sos_model(get_sos_model)
        expected = get_sos_model['narratives'][0]
        # read one
        assert store.read_narrative(get_sos_model['name'], expected['name']) == expected

    def test_strategies(self, store, strategies):
        model_run_name = 'test_modelrun'
        # write
        store.write_strategies(model_run_name, strategies)
        # read
        assert store.read_strategies(model_run_name) == strategies


class TestStoreMetadata():
    def test_units(self, store, unit_definitions):
        # write
        store.write_unit_definitions(unit_definitions)
        # read
        assert store.read_unit_definitions() == unit_definitions

    def test_dimensions(self, store, dimension):
        # write
        store.write_dimension(dimension)
        # read all
        assert store.read_dimensions() == [dimension]
        # read one
        assert store.read_dimension(dimension['name']) == dimension
        # update
        store.update_dimension(dimension['name'], dimension)
        # delete
        store.delete_dimension(dimension['name'])
        assert store.read_dimensions() == []


class TestStoreData():
    def test_scenario_variant_data(self, store, sample_dimensions, scenario,
                                   sample_scenario_data):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_scenario(scenario)
        # pick out single sample
        key = next(iter(sample_scenario_data))
        scenario_name, variant_name, variable = key
        scenario_variant_data = sample_scenario_data[key]
        # write
        store.write_scenario_variant_data(
            scenario_name, variant_name, scenario_variant_data
        )
        # read
        actual = store.read_scenario_variant_data(
            scenario_name, variant_name, variable
        )
        assert actual == scenario_variant_data

    def test_narrative_variant_data(self, store, sample_dimensions, get_sos_model,
                                    get_sector_model, energy_supply_sector_model,
                                    sample_narrative_data):
        # Setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_sos_model(get_sos_model)
        store.write_model(get_sector_model)
        store.write_model(energy_supply_sector_model)
        # pick out single sample
        key = (
            'energy',
            'technology',
            'high_tech_dsm',
            'smart_meter_savings'
        )
        sos_model_name, narrative_name, variant_name, param_name = key
        narrative_variant_data = sample_narrative_data[key]
        # write
        store.write_narrative_variant_data(
            sos_model_name, narrative_name, variant_name, narrative_variant_data)
        # read
        actual = store.read_narrative_variant_data(
            sos_model_name, narrative_name, variant_name, param_name)
        assert actual == narrative_variant_data

    def test_model_parameter_default(self, store, get_multidimensional_param,
                                     get_sector_model, sample_dimensions):
        param_data = get_multidimensional_param
        for dim in sample_dimensions:
            store.write_dimension(dim)
        for dim in param_data.dims:
            store.write_dimension({'name': dim, 'elements': param_data.dim_elements(dim)})
        get_sector_model['parameters'] = [param_data.spec.as_dict()]
        store.write_model(get_sector_model)
        # write
        store.write_model_parameter_default(
            get_sector_model['name'], param_data.name, param_data)
        # read
        actual = store.read_model_parameter_default(get_sector_model['name'], param_data.name)
        assert actual == param_data

    def test_interventions(self, store, sample_dimensions, get_sector_model, interventions):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_model(get_sector_model)
        # write
        store.write_interventions(get_sector_model['name'], interventions)
        # read
        assert store.read_interventions(get_sector_model['name']) == interventions

    def test_initial_conditions(self, store, sample_dimensions, initial_conditions,
                                get_sos_model, get_sector_model, energy_supply_sector_model,
                                minimal_model_run):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_sos_model(get_sos_model)
        store.write_model_run(minimal_model_run)
        store.write_model(get_sector_model)
        store.write_model(energy_supply_sector_model)
        # write
        store.write_initial_conditions(get_sector_model['name'], initial_conditions)
        # read
        assert store.read_initial_conditions(get_sector_model['name']) == initial_conditions
        # read all for a model run
        actual = store.read_all_initial_conditions(minimal_model_run['name'])
        assert actual == initial_conditions

    def test_state(self, store, state):
        # write
        store.write_state(state, 'model_run_name', 0, 0)
        # read
        assert store.read_state('model_run_name', 0, 0) == state

    def test_conversion_coefficients(self, store, conversion_coefficients,
                                     conversion_source_spec, conversion_sink_spec):
        # write
        store.write_coefficients(
            conversion_source_spec, conversion_sink_spec, conversion_coefficients)
        # read
        numpy.testing.assert_equal(
            store.read_coefficients(conversion_source_spec, conversion_sink_spec),
            conversion_coefficients
        )

    def test_results(self, store, sample_results):
        # write
        store.write_results(sample_results, 'model_run_name', 'model_name', 0)
        # read
        spec = sample_results.spec
        assert store.read_results('model_run_name', 'model_name', spec, 0) == sample_results
        # check
        assert store.available_results('model_run_name') == \
            [(0, None, 'model_name', spec.name)]

    def test_warm_start(self, store, sample_results):
        assert store.prepare_warm_start('test_model_run') is None
        timestep = 2020
        store.write_results(sample_results, 'test_model_run', 'model_name', timestep)
        assert store.prepare_warm_start('test_model_run') == timestep
