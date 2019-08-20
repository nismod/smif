"""Test the Store interface

Many methods simply proxy to config/metadata/data store implementations, but there is some
cross-coordination and there are some convenience methods implemented at this layer.
"""
import os

import numpy as np
import numpy.testing
from pytest import fixture, raises
from smif.data_layer import Store
from smif.data_layer.data_array import DataArray
from smif.data_layer.memory_interface import (MemoryConfigStore,
                                              MemoryDataStore,
                                              MemoryMetadataStore)
from smif.exception import SmifDataError, SmifDataNotFoundError
from smif.metadata import Spec


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


@fixture
def full_store(store, get_sos_model, get_sector_model, energy_supply_sector_model,
               sample_scenarios, model_run, sample_dimensions):
    for dim in sample_dimensions:
        store.write_dimension(dim)
    store.write_model(get_sector_model)
    store.write_model(energy_supply_sector_model)
    for scenario in sample_scenarios:
        store.write_scenario(scenario)
    store.write_sos_model(get_sos_model)
    store.write_model_run(model_run)
    return store


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
        assert store.read_model(get_sector_model['name'],
                                skip_coords=True) == get_sector_model_no_coords
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
        assert store.read_scenario(scenario['name'], skip_coords=True) == scenario_no_coords
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

    def test_prepare_scenario(self, store, scenario, scenario_2_variants,
                              scenario_no_variant, sample_dimensions):
        for dim in sample_dimensions:
            store.write_dimension(dim)
        # Insert template_scenario dict in underlying
        # MemoryConfigStore
        store.write_scenario(scenario)
        store.write_scenario(scenario_2_variants)
        store.write_scenario(scenario_no_variant)

        list_of_variants = range(1, 4)

        # Must raise exception if scenario defines > 1 variants
        with raises(SmifDataError) as ex:
            store.prepare_scenario(scenario_2_variants['name'], list_of_variants)
        assert "must define one unique template variant" in str(ex.value)

        # Must raise exception if scenario defines 0 variants
        with raises(SmifDataError) as ex:
            store.prepare_scenario(scenario_no_variant['name'], list_of_variants)
        assert "must define one unique template variant" in str(ex.value)

        store.prepare_scenario(scenario['name'], list_of_variants)

        updated_scenario = store.read_scenario(scenario['name'])

        assert len(updated_scenario['variants']) == 3
        assert updated_scenario['variants'][0]['name'] == 'mortality_001'
        new_variant = store.read_scenario_variant(scenario['name'], 'mortality_001')
        #
        assert new_variant['name'] == 'mortality_001'
        assert new_variant['description'] == 'mortality variant number 001'
        assert new_variant['data']['mortality'] == 'mortality_low001.csv'

        assert updated_scenario['variants'][1]['name'] == 'mortality_002'
        new_variant = store.read_scenario_variant(scenario['name'], 'mortality_002')
        assert new_variant['name'] == 'mortality_002'
        assert new_variant['description'] == 'mortality variant number 002'
        assert new_variant['data']['mortality'] == 'mortality_low002.csv'

        assert updated_scenario['variants'][2]['name'] == 'mortality_003'
        new_variant = store.read_scenario_variant(scenario['name'], 'mortality_003')
        assert new_variant['name'] == 'mortality_003'
        assert new_variant['description'] == 'mortality variant number 003'
        assert new_variant['data']['mortality'] == 'mortality_low003.csv'

    def test_prepare_model_runs(self, store, model_run, sample_scenarios, sample_dimensions):
        for dim in sample_dimensions:
            store.write_dimension(dim)
        scenario = sample_scenarios[0]
        store.write_model_run(model_run)
        store.write_strategies(model_run['name'], model_run['strategies'])
        store.write_scenario(scenario)

        # Generate 2 model runs for variants Low and High
        store.prepare_model_runs(model_run['name'], scenario['name'], 0, 1)
        list_of_mr = store.read_model_runs()
        assert len(list_of_mr) == 3
        assert list_of_mr[0] == model_run
        assert list_of_mr[1]['name'] == model_run['name'] + '_' + scenario['variants'][0][
            'name']
        assert list_of_mr[2]['name'] == model_run['name'] + '_' + scenario['variants'][1][
            'name']
        store.delete_model_run(list_of_mr[1]['name'])
        store.delete_model_run(list_of_mr[2]['name'])
        # Generate only one model run for variant Low
        store.prepare_model_runs(model_run['name'], scenario['name'], 0, 0)
        list_of_mr = store.read_model_runs()
        assert len(list_of_mr) == 2
        assert list_of_mr[0] == model_run
        assert list_of_mr[1]['name'] == model_run['name'] + '_' + scenario['variants'][0][
            'name']

        # Tidy up batch file
        os.remove('{}.batch'.format(model_run['name']))

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
    @fixture(scope='function')
    def setup(self, store, sample_dimensions, scenario,
              sample_scenario_data):
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
        return key, scenario_variant_data

    def test_convert_strategies_data(self, empty_store, store, strategies):
        src_store = store
        tgt_store = empty_store  # Store with target data format
        model_run_name = 'test_modelrun'
        # write
        src_store.write_strategies(model_run_name, strategies)
        # convert
        src_store.convert_strategies_data(model_run_name, tgt_store)
        # assert
        for strategy in strategies:
            if 'interventions' in strategy:  # If the stategy fixture defines interventions
                expected = src_store.read_strategy_interventions(strategy)
                assert expected == tgt_store.read_strategy_interventions(strategy)

    def test_scenario_variant_data(self, store,
                                   setup):
        # The sample_scenario_data fixture provides data with a spec including timestep
        # dimension containing a single coordinate of 2015. Note the asymmetry in the write
        # and read methods here: writing requires the full DataArray object with the full
        # spec including timestep, but the reading requires a specific timestep to be supplied.
        # The data read back in, therefore, has lower dimensionality.

        key, scenario_variant_data = setup
        scenario_name, variant_name, variable = key

        assert store.read_scenario_variant_data(scenario_name, variant_name, variable,
                                                2015, assert_exists=True)
        # Read 2015
        actual = store.read_scenario_variant_data(
            scenario_name, variant_name, variable, 2015
        )
        assert (actual.data == scenario_variant_data.data[0]).all()

        # Read 2016
        actual = store.read_scenario_variant_data(
            scenario_name, variant_name, variable, 2016
        )
        assert (actual.data == scenario_variant_data.data[1]).all()

    def test_convert_scenario_data(self, empty_store, store, sample_dimensions, scenario,
                                   sample_scenario_data, model_run):
        src_store = store
        tgt_store = empty_store  # Store with target data format
        # setup
        model_run['scenarios'] = {'mortality': 'low'}
        model_run['timesteps'] = [2015, 2016]
        src_store.write_model_run(model_run)
        tgt_store.write_model_run(model_run)
        for dim in sample_dimensions:
            src_store.write_dimension(dim)
            tgt_store.write_dimension(dim)

        src_store.write_scenario(scenario)
        tgt_store.write_scenario(scenario)
        # pick out single sample
        key = next(iter(sample_scenario_data))
        scenario_name, variant_name, variable = key
        scenario_variant_data = sample_scenario_data[key]
        # write
        src_store.write_scenario_variant_data(
            scenario_name, variant_name, scenario_variant_data
        )
        src_store.convert_scenario_data(model_run['name'], tgt_store)

        for variant in src_store.read_scenario_variants(scenario_name):
            for variable in variant['data']:
                expected = src_store.read_scenario_variant_data(
                    scenario_name, variant['name'], variable, timesteps=model_run['timesteps'])
                result = tgt_store.read_scenario_variant_data(
                    scenario_name, variant['name'], variable, timesteps=model_run['timesteps'])
                assert result == expected

    def test_scenario_variant_data_mult_one_year(self, store, setup):
        key, scenario_variant_data = setup
        scenario_name, variant_name, variable = key

        actual = store.read_scenario_variant_data(
            scenario_name, variant_name, variable, timesteps=[2016])

        assert (actual.data == [scenario_variant_data.data[1]]).all()

    def test_scenario_variant_data_mult_mult_years(self, store, setup):

        key, scenario_variant_data = setup
        scenario_name, variant_name, variable = key

        actual = store.read_scenario_variant_data(
            scenario_name, variant_name, variable, timesteps=[2015, 2016])

        assert (actual.data == scenario_variant_data.data).all()

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

        assert store.read_narrative_variant_data(
            sos_model_name, narrative_name, variant_name, param_name, assert_exists=True)
        # read
        actual = store.read_narrative_variant_data(
            sos_model_name, narrative_name, variant_name, param_name)
        assert actual == narrative_variant_data

    def test_convert_narrative_data(self, empty_store, store, sample_dimensions, get_sos_model,
                                    get_sector_model, energy_supply_sector_model,
                                    sample_narrative_data, model_run):
        src_store = store
        tgt_store = empty_store
        # Setup
        for dim in sample_dimensions:
            src_store.write_dimension(dim)
            tgt_store.write_dimension(dim)
        src_store.write_sos_model(get_sos_model)
        src_store.write_model(get_sector_model)
        src_store.write_model(energy_supply_sector_model)
        tgt_store.write_sos_model(get_sos_model)
        tgt_store.write_model(get_sector_model)
        tgt_store.write_model(energy_supply_sector_model)

        for narrative in get_sos_model['narratives']:
            for variant in narrative['variants']:
                for param_name in variant['data']:
                    key = (
                        get_sos_model['name'],
                        narrative['name'],
                        variant['name'],
                        param_name
                    )
                    narrative_variant_data = sample_narrative_data[key]
                    # write
                    src_store.write_narrative_variant_data(
                        get_sos_model['name'], narrative['name'], variant['name'],
                        narrative_variant_data)

        src_store.convert_narrative_data(model_run['sos_model'], tgt_store)

        for narrative in get_sos_model['narratives']:
            for variant in narrative['variants']:
                for param_name in variant['data']:
                    expected = src_store.read_narrative_variant_data(
                        get_sos_model['name'], narrative['name'], variant['name'], param_name)
                    result = tgt_store.read_narrative_variant_data(
                        get_sos_model['name'], narrative['name'], variant['name'], param_name)
                    assert result == expected

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
        assert store.read_model_parameter_default(get_sector_model['name'],
                                                  param_data.name, assert_exists=True)
        # read
        actual = store.read_model_parameter_default(get_sector_model['name'], param_data.name)
        assert actual == param_data

    def test_convert_model_parameter_default_data(self, empty_store, store,
                                                  get_multidimensional_param,
                                                  get_sector_model, sample_dimensions):
        src_store = store
        tgt_store = empty_store
        param_data = get_multidimensional_param
        for store in [src_store, tgt_store]:
            for dim in sample_dimensions:
                store.write_dimension(dim)
            for dim in param_data.dims:
                store.write_dimension({'name': dim, 'elements': param_data.dim_elements(dim)})

            get_sector_model['parameters'] = [param_data.spec.as_dict()]
            store.write_model(get_sector_model)

        # write
        src_store.write_model_parameter_default(
            get_sector_model['name'], param_data.name, param_data)
        # convert
        src_store.convert_model_parameter_default_data(get_sector_model['name'], tgt_store)

        expected = src_store.read_model_parameter_default(
            get_sector_model['name'], param_data.name)
        result = tgt_store.read_model_parameter_default(
            get_sector_model['name'], param_data.name)

        assert result == expected

    def test_interventions(self, store, sample_dimensions, get_sector_model, interventions):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_model(get_sector_model)
        # write
        store.write_interventions(get_sector_model['name'], interventions)
        # read
        assert store.read_interventions(get_sector_model['name']) == interventions

    def test_convert_interventions_data(self, empty_store, store, sample_dimensions,
                                        get_sector_model, interventions):
        src_store = store
        tgt_store = empty_store
        get_sector_model['interventions'] = ['energy_demand.csv']
        # setup
        for dim in sample_dimensions:
            src_store.write_dimension(dim)
            tgt_store.write_dimension(dim)
        src_store.write_model(get_sector_model)
        tgt_store.write_model(get_sector_model)
        # write
        src_store.write_interventions(get_sector_model['name'], interventions)

        src_store.convert_interventions_data(get_sector_model['name'], tgt_store)

        assert interventions == tgt_store.read_interventions(get_sector_model['name'])

    def test_read_write_interventions_file(self, store, sample_dimensions,
                                           get_sector_model, interventions):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        get_sector_model['interventions'] = ['path']
        store.write_model(get_sector_model)
        # write
        store.write_interventions_file(get_sector_model['name'], 'path', interventions)
        # check data existence
        assert store.read_interventions_file(
            get_sector_model['name'], 'path', assert_exists=True)

        result = store.read_interventions_file(get_sector_model['name'], 'path')
        assert result == interventions

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

    def test_convert_initial_conditions_data(self, empty_store, store, sample_dimensions,
                                             initial_conditions, get_sos_model,
                                             get_sector_model, energy_supply_sector_model,
                                             minimal_model_run):
        src_store = store
        tgt_store = empty_store
        get_sector_model['initial_conditions'] = ['energy_demand.csv']
        for store in [src_store, tgt_store]:
            for dim in sample_dimensions:
                store.write_dimension(dim)
            store.write_sos_model(get_sos_model)
            store.write_model_run(minimal_model_run)
            store.write_model(get_sector_model)
            store.write_model(energy_supply_sector_model)

        src_store.write_initial_conditions(get_sector_model['name'], initial_conditions)

        src_store.convert_initial_conditions_data(get_sector_model['name'], tgt_store)

        assert initial_conditions == tgt_store.read_initial_conditions(
            get_sector_model['name'])

    def test_read_write_initial_conditions_file(self, store, sample_dimensions,
                                                get_sector_model, initial_conditions):
        # setup
        for dim in sample_dimensions:
            store.write_dimension(dim)
        get_sector_model['initial_conditions'] = ['path']
        store.write_model(get_sector_model)
        # write
        store.write_initial_conditions_file(get_sector_model['name'],
                                            'path', initial_conditions)
        assert store.read_initial_conditions_file(get_sector_model['name'], 'path',
                                                  assert_exists=True)
        result = store.read_initial_conditions_file(get_sector_model['name'], 'path')

        assert result == initial_conditions

    def test_state(self, store, state):
        # write
        store.write_state(state, 'model_run_name', 0, 0)
        # read
        assert store.read_state('model_run_name', 0, 0) == state

    def test_conversion_coefficients(self, store, conversion_coefficients):
        # write
        store.write_coefficients(
            'source_dim', 'sink_dim', conversion_coefficients)
        # read
        numpy.testing.assert_equal(
            store.read_coefficients('source_dim', 'sink_dim'),
            conversion_coefficients
        )

    def test_results(self, store, sample_results):
        # write
        store.write_results(sample_results, 'model_run_name', 'model_name', 0)
        # read
        spec = sample_results.spec
        assert store.read_results('model_run_name', 'model_name', spec, 0) == sample_results
        # check
        assert store.available_results('model_run_name') == [
            (0, None, 'model_name', spec.name)
        ]
        # delete
        store.clear_results('model_run_name')
        assert not store.available_results('model_run_name')

    def test_no_completed_jobs(self, full_store):
        expected = []
        actual = full_store.completed_jobs('unique_model_run_name')
        assert actual == expected

    def test_expected_model_outputs(self, full_store):
        actual = full_store.expected_model_outputs('unique_model_run_name')
        expected = [('energy_demand', 'gas_demand')]
        assert actual == expected

    def test_some_completed_jobs(self, full_store, sample_gas_demand_results):
        expected = [
            (2015, 0, 'energy_demand'),
            (2020, 0, 'energy_demand'),
        ]
        full_store.write_results(
            sample_gas_demand_results, 'unique_model_run_name', 'energy_demand', 2015, 0)
        full_store.write_results(
            sample_gas_demand_results, 'unique_model_run_name', 'energy_demand', 2020, 0)
        actual = full_store.completed_jobs('unique_model_run_name')
        assert actual == expected

    def test_filter_complete_available_results(self, store):
        available_results = [
            (2020, 0, 'test_model', 'output_a'),
            (2020, 0, 'test_model', 'output_b'),
            (2025, 0, 'test_model', 'output_a'),
            (2030, 0, 'other_model', 'output_other'),
        ]
        model_outputs = [
            ('test_model', 'output_a'),
            ('test_model', 'output_b'),
        ]
        expected = [
            (2020, 0, 'test_model')
        ]
        actual = store.filter_complete_available_results(available_results, model_outputs)
        assert actual == expected

    def test_warm_start(self, store, sample_results):
        assert store.prepare_warm_start('test_model_run') is None
        timestep = 2020
        store.write_results(sample_results, 'test_model_run', 'model_name', timestep)
        assert store.prepare_warm_start('test_model_run') == timestep

    def test_canonical_available_results(self, store, sample_results):

        store.write_results(sample_results, 'model_run_name', 'model_name', 2010, 0)
        store.write_results(sample_results, 'model_run_name', 'model_name', 2015, 0)
        store.write_results(sample_results, 'model_run_name', 'model_name', 2010, 1)
        store.write_results(sample_results, 'model_run_name', 'model_name', 2015, 1)
        store.write_results(sample_results, 'model_run_name', 'model_name', 2020, 1)

        output_name = sample_results.spec.name

        correct_results = set()
        correct_results.add((2010, 0, 'model_name', output_name))
        correct_results.add((2015, 0, 'model_name', output_name))
        correct_results.add((2020, 0, 'model_name', output_name))

        assert (store.canonical_available_results('model_run_name') == correct_results)

    def test_canonical_expected_results(
            self, store, sample_dimensions, get_sos_model, get_sector_model,
            energy_supply_sector_model, model_run
    ):

        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_sos_model(get_sos_model)
        store.write_model_run(model_run)
        store.write_model(get_sector_model)
        store.write_model(energy_supply_sector_model)

        correct_results = set()
        correct_results.add((2015, 0, 'energy_demand', 'gas_demand'))
        correct_results.add((2020, 0, 'energy_demand', 'gas_demand'))
        correct_results.add((2025, 0, 'energy_demand', 'gas_demand'))

        assert (store.canonical_expected_results(model_run['name']) == correct_results)

    def test_canonical_missing_results(
            self, store, sample_dimensions, get_sos_model, get_sector_model,
            energy_supply_sector_model, model_run
    ):

        for dim in sample_dimensions:
            store.write_dimension(dim)
        store.write_sos_model(get_sos_model)
        store.write_model_run(model_run)
        store.write_model(get_sector_model)
        store.write_model(energy_supply_sector_model)

        # All the results are missing
        missing_results = set()
        missing_results.add((2015, 0, 'energy_demand', 'gas_demand'))
        missing_results.add((2020, 0, 'energy_demand', 'gas_demand'))
        missing_results.add((2025, 0, 'energy_demand', 'gas_demand'))

        assert (store.canonical_missing_results(model_run['name']) == missing_results)

        spec = Spec(name='gas_demand', dtype='float')
        data = np.array(1, dtype=float)
        fake_data = DataArray(spec, data)

        store.write_results(fake_data, model_run['name'], 'energy_demand', 2015, 0)
        missing_results.remove((2015, 0, 'energy_demand', 'gas_demand'))

        assert (store.canonical_missing_results(model_run['name']) == missing_results)

    def test_get_results(self):
        # This is difficult to test without fixtures defining an entire canonical project.
        # See smif issue #304 (https://github.com/nismod/smif/issues/304).
        # Todo: mock a store with known results that can be obtained with get_results(...)
        # This requires a model run with sector model, and a sector model with valid inputs and
        # outputs, and results with valid spec, etc. Some of this functionality exists in
        # fixtures provided in `conftest.py`.
        pass


class TestWrongRaises:

    def test_narrative_variant(self, store, sample_dimensions,
                               get_sos_model, get_sector_model,
                               energy_supply_sector_model,
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

        with raises(SmifDataNotFoundError) as ex:
            store.read_narrative_variant_data(
                sos_model_name, narrative_name, 'bla', param_name)

        expected = "Variant name 'bla' does not exist in narrative 'technology'"

        assert expected in str(ex.value)

    def test_narrative_name(self, store, sample_dimensions, get_sos_model,
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

        with raises(SmifDataNotFoundError) as ex:
            store.read_narrative_variant_data(
                sos_model_name, 'bla', variant_name, param_name)

        expected = "Narrative name 'bla' does not exist in sos_model 'energy'"
        assert expected in str(ex.value)
