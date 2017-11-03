"""Test data interface
"""
import csv
import json
import os
from copy import copy, deepcopy
from pytest import raises

from smif.data_layer import (DataExistsError, DataMismatchError, DataNotFoundError)
from smif.data_layer.datafile_interface import transform_leaves
from smif.data_layer.load import dump


class TestDatafileInterface():

    def test_sos_model_run_read_all(self, get_sos_model_run, get_handler):
        """Test to write two sos_model_run configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """
        config_handler = get_handler

        sos_model_run1 = get_sos_model_run
        sos_model_run1['name'] = 'sos_model_run1'
        config_handler.write_sos_model_run(sos_model_run1)

        sos_model_run2 = get_sos_model_run
        sos_model_run2['name'] = 'sos_model_run2'
        config_handler.write_sos_model_run(sos_model_run2)

        sos_model_runs = config_handler.read_sos_model_runs()
        sos_model_run_names = list(sos_model_run['name'] for sos_model_run in sos_model_runs)

        assert 'sos_model_run1' in sos_model_run_names
        assert 'sos_model_run2' in sos_model_run_names
        assert len(sos_model_runs) == 2

    def test_sos_model_run_write_twice(self, get_sos_model_run, get_handler):
        """Test that writing a sos_model_run should fail (not overwrite).
        """
        config_handler = get_handler

        sos_model_run1 = get_sos_model_run
        sos_model_run1['name'] = 'unique'
        config_handler.write_sos_model_run(sos_model_run1)

        with raises(DataExistsError) as ex:
            config_handler.write_sos_model_run(sos_model_run1)
        assert "sos_model_run 'unique' already exists" in str(ex)

    def test_sos_model_run_read_one(self, get_sos_model_run, get_handler):
        """Test reading a single sos_model_run.
        """
        config_handler = get_handler

        sos_model_run1 = get_sos_model_run
        sos_model_run1['name'] = 'sos_model_run1'
        config_handler.write_sos_model_run(sos_model_run1)

        sos_model_run2 = get_sos_model_run
        sos_model_run2['name'] = 'sos_model_run2'
        config_handler.write_sos_model_run(sos_model_run2)

        sos_model_run = config_handler.read_sos_model_run('sos_model_run2')
        assert sos_model_run['name'] == 'sos_model_run2'

    def test_sos_model_run_read_missing(self, get_handler):
        """Test that reading a missing sos_model_run fails.
        """
        config_handler = get_handler
        with raises(DataNotFoundError) as ex:
            config_handler.read_sos_model_run('missing_name')
        assert "sos_model_run 'missing_name' not found" in str(ex)

    def test_sos_model_run_update(self, get_sos_model_run, get_handler):
        """Test updating a sos_model_run description
        """
        config_handler = get_handler
        sos_model_run = get_sos_model_run
        sos_model_run['name'] = 'to_update'
        sos_model_run['description'] = 'before'

        config_handler.write_sos_model_run(sos_model_run)

        sos_model_run['description'] = 'after'
        config_handler.update_sos_model_run('to_update', sos_model_run)

        actual = config_handler.read_sos_model_run('to_update')
        assert actual['description'] == 'after'

    def test_sos_model_run_update_mismatch(self, get_sos_model_run, get_handler):
        """Test that updating a sos_model_run with mismatched name should fail
        """
        config_handler = get_handler
        sos_model_run = get_sos_model_run

        sos_model_run['name'] = 'sos_model_run'
        with raises(DataMismatchError) as ex:
            config_handler.update_sos_model_run('sos_model_run2', sos_model_run)
        assert "name 'sos_model_run2' must match 'sos_model_run'" in str(ex)

    def test_sos_model_run_update_missing(self, get_sos_model_run, get_handler):
        """Test that updating a nonexistent sos_model_run should fail
        """
        config_handler = get_handler
        sos_model_run = get_sos_model_run
        sos_model_run['name'] = 'missing_name'

        with raises(DataNotFoundError) as ex:
            config_handler.update_sos_model_run('missing_name', sos_model_run)
        assert "sos_model_run 'missing_name' not found" in str(ex)

    def test_sos_model_run_delete(self, get_sos_model_run, get_handler):
        """Test that updating a nonexistent sos_model_run should fail
        """
        config_handler = get_handler
        sos_model_run = get_sos_model_run
        sos_model_run['name'] = 'to_delete'

        config_handler.write_sos_model_run(sos_model_run)
        before_delete = config_handler.read_sos_model_runs()
        assert len(before_delete) == 1

        config_handler.delete_sos_model_run('to_delete')
        after_delete = config_handler.read_sos_model_runs()
        assert len(after_delete) == 0

    def test_sos_model_run_delete_missing(self, get_sos_model_run, get_handler):
        """Test that updating a nonexistent sos_model_run should fail
        """
        config_handler = get_handler
        with raises(DataNotFoundError) as ex:
            config_handler.delete_sos_model_run('missing_name')
        assert "sos_model_run 'missing_name' not found" in str(ex)

    def test_sos_model_read_all(self, get_sos_model, get_handler):
        """Test to write two sos_model configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """
        config_handler = get_handler

        sos_model1 = get_sos_model
        sos_model1['name'] = 'sos_model1'
        config_handler.write_sos_model(sos_model1)

        sos_model2 = get_sos_model
        sos_model2['name'] = 'sos_model2'
        config_handler.write_sos_model(sos_model2)

        sos_models = config_handler.read_sos_models()
        sos_model_names = list(sos_model['name'] for sos_model in sos_models)

        assert 'sos_model1' in sos_model_names
        assert 'sos_model2' in sos_model_names
        assert len(sos_models) == 2

    def test_sos_model_write_twice(self, get_sos_model, get_handler):
        """Test that writing a sos_model should fail (not overwrite).
        """
        config_handler = get_handler

        sos_model1 = get_sos_model
        sos_model1['name'] = 'unique'
        config_handler.write_sos_model(sos_model1)

        with raises(DataExistsError) as ex:
            config_handler.write_sos_model(sos_model1)
        assert "sos_model 'unique' already exists" in str(ex)

    def test_sos_model_read_one(self, get_sos_model, get_handler):
        """Test reading a single sos_model.
        """
        config_handler = get_handler

        sos_model1 = get_sos_model
        sos_model1['name'] = 'sos_model1'
        config_handler.write_sos_model(sos_model1)

        sos_model2 = get_sos_model
        sos_model2['name'] = 'sos_model2'
        config_handler.write_sos_model(sos_model2)

        sos_model = config_handler.read_sos_model('sos_model2')
        assert sos_model['name'] == 'sos_model2'

    def test_sos_model_read_missing(self, get_handler):
        """Test that reading a missing sos_model fails.
        """
        config_handler = get_handler
        with raises(DataNotFoundError) as ex:
            config_handler.read_sos_model('missing_name')
        assert "sos_model 'missing_name' not found" in str(ex)

    def test_sos_model_update(self, get_sos_model, get_handler):
        """Test updating a sos_model description
        """
        config_handler = get_handler
        sos_model = get_sos_model
        sos_model['name'] = 'to_update'
        sos_model['description'] = 'before'

        config_handler.write_sos_model(sos_model)

        sos_model['description'] = 'after'
        config_handler.update_sos_model('to_update', sos_model)

        actual = config_handler.read_sos_model('to_update')
        assert actual['description'] == 'after'

    def test_sos_model_update_mismatch(self, get_sos_model, get_handler):
        """Test that updating a sos_model with mismatched name should fail
        """
        config_handler = get_handler
        sos_model = get_sos_model

        sos_model['name'] = 'sos_model'
        with raises(DataMismatchError) as ex:
            config_handler.update_sos_model('sos_model2', sos_model)
        assert "name 'sos_model2' must match 'sos_model'" in str(ex)

    def test_sos_model_update_missing(self, get_sos_model, get_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        config_handler = get_handler
        sos_model = get_sos_model
        sos_model['name'] = 'missing_name'

        with raises(DataNotFoundError) as ex:
            config_handler.update_sos_model('missing_name', sos_model)
        assert "sos_model 'missing_name' not found" in str(ex)

    def test_sos_model_delete(self, get_sos_model, get_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        config_handler = get_handler
        sos_model = get_sos_model
        sos_model['name'] = 'to_delete'

        config_handler.write_sos_model(sos_model)
        before_delete = config_handler.read_sos_models()
        assert len(before_delete) == 1

        config_handler.delete_sos_model('to_delete')
        after_delete = config_handler.read_sos_models()
        assert len(after_delete) == 0

    def test_sos_model_delete_missing(self, get_sos_model, get_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        config_handler = get_handler
        with raises(DataNotFoundError) as ex:
            config_handler.delete_sos_model('missing_name')
        assert "sos_model 'missing_name' not found" in str(ex)

    def test_sector_model_read_all(self, get_sector_model, get_handler):
        """Test to write two sector_model configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """
        config_handler = get_handler

        sector_model1 = get_sector_model
        sector_model1['name'] = 'sector_model1'
        config_handler.write_sector_model(sector_model1)

        sector_model2 = get_sector_model
        sector_model2['name'] = 'sector_model2'
        config_handler.write_sector_model(sector_model2)

        sector_models = config_handler.read_sector_models()
        sector_model_names = list(sector_model['name'] for sector_model in sector_models)

        assert 'sector_model1' in sector_model_names
        assert 'sector_model2' in sector_model_names
        assert len(sector_models) == 2

    def test_sector_model_write_twice(self, get_sector_model, get_handler):
        """Test that writing a sector_model should fail (not overwrite).
        """
        config_handler = get_handler

        sector_model1 = get_sector_model
        sector_model1['name'] = 'unique'
        config_handler.write_sector_model(sector_model1)

        with raises(DataExistsError) as ex:
            config_handler.write_sector_model(sector_model1)
        assert "sector_model 'unique' already exists" in str(ex)

    def test_sector_model_read_one(self, get_sector_model, get_handler):
        """Test reading a single sector_model.
        """
        config_handler = get_handler

        sector_model1 = get_sector_model
        sector_model1['name'] = 'sector_model1'
        config_handler.write_sector_model(sector_model1)

        sector_model2 = get_sector_model
        sector_model2['name'] = 'sector_model2'
        config_handler.write_sector_model(sector_model2)

        sector_model = config_handler.read_sector_model('sector_model2')
        assert sector_model['name'] == 'sector_model2'

    def test_sector_model_read_missing(self, get_handler):
        """Test that reading a missing sector_model fails.
        """
        config_handler = get_handler
        with raises(DataNotFoundError) as ex:
            config_handler.read_sector_model('missing_name')
        assert "sector_model 'missing_name' not found" in str(ex)

    def test_sector_model_update(self, get_sector_model, get_handler):
        """Test updating a sector_model description
        """
        config_handler = get_handler
        sector_model = get_sector_model
        sector_model['name'] = 'to_update'
        sector_model['description'] = 'before'

        config_handler.write_sector_model(sector_model)

        sector_model['description'] = 'after'
        config_handler.update_sector_model('to_update', sector_model)

        actual = config_handler.read_sector_model('to_update')
        assert actual['description'] == 'after'

    def test_sector_model_update_mismatch(self, get_sector_model, get_handler):
        """Test that updating a sector_model with mismatched name should fail
        """
        config_handler = get_handler
        sector_model = get_sector_model

        sector_model['name'] = 'sector_model'
        with raises(DataMismatchError) as ex:
            config_handler.update_sector_model('sector_model2', sector_model)
        assert "name 'sector_model2' must match 'sector_model'" in str(ex)

    def test_sector_model_update_missing(self, get_sector_model, get_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        config_handler = get_handler
        sector_model = get_sector_model
        sector_model['name'] = 'missing_name'

        with raises(DataNotFoundError) as ex:
            config_handler.update_sector_model('missing_name', sector_model)
        assert "sector_model 'missing_name' not found" in str(ex)

    def test_sector_model_delete(self, get_sector_model, get_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        config_handler = get_handler
        sector_model = get_sector_model
        sector_model['name'] = 'to_delete'

        config_handler.write_sector_model(sector_model)
        before_delete = config_handler.read_sector_models()
        assert len(before_delete) == 1

        config_handler.delete_sector_model('to_delete')
        after_delete = config_handler.read_sector_models()
        assert len(after_delete) == 0

    def test_sector_model_delete_missing(self, get_sector_model, get_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        config_handler = get_handler
        with raises(DataNotFoundError) as ex:
            config_handler.delete_sector_model('missing_name')
        assert "sector_model 'missing_name' not found" in str(ex)

    def test_region_definition_data(self, setup_folder_structure, setup_region_data,
                                    get_handler):
        """ Test to dump a region_definition_set (GeoJSON) data-file and then read the data
        using the datafile interface. Finally check if the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        region_definition_data = setup_region_data

        with open(os.path.join(str(basefolder), 'data', 'region_definitions',
                               'test_region.json'), 'w+') as region_definition_file:
            json.dump(region_definition_data, region_definition_file)

        config_handler = get_handler
        test_region_definition = config_handler.read_region_definition_data(
            'lad')

        assert test_region_definition[0]['properties']['name'] == 'oxford'

    def test_scenario_data(self, setup_folder_structure, get_handler,
                           get_scenario_data):
        """ Test to dump a scenario (CSV) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        scenario_data = get_scenario_data

        keys = scenario_data[0].keys()
        with open(os.path.join(str(basefolder), 'data', 'scenarios',
                               'population_high.csv'), 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        config_handler = get_handler
        test_scenario = config_handler.read_scenario_data(
            'High Population (ONS)')

        assert len(test_scenario) == 1
        assert 'population_count' in test_scenario
        assert test_scenario['population_count'][0]['region'] == 'GB'

    def test_narrative_data(self, setup_folder_structure, get_handler, get_narrative_data):
        """ Test to dump a narrative (yml) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        narrative_data_path = os.path.join(str(basefolder), 'data', 'narratives',
                                           'central_planning.yml')
        dump(get_narrative_data, narrative_data_path)

        config_handler = get_handler
        test_narrative = config_handler.read_narrative_data('Central Planning')

        assert test_narrative[0]['energy_demand'][0]['name'] == 'smart_meter_savings'

    def test_project_region_definitions(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # region_definition sets / read existing (from fixture)
        region_definitions = config_handler.read_region_definitions()
        assert region_definitions[0]['name'] == 'lad'
        assert len(region_definitions) == 1

        # region_definition sets / add
        region_definition = {
            'name': 'lad_NL',
            'description': 'Local authority districts for the Netherlands',
            'filename': 'lad_NL.csv'
        }
        config_handler.write_region_definition(region_definition)
        region_definitions = config_handler.read_region_definitions()
        assert len(region_definitions) == 2
        for region_definition in region_definitions:
            if region_definition['name'] == 'lad_NL':
                assert region_definition['filename'] == 'lad_NL.csv'

        # region_definition sets / modify
        region_definition = {
            'name': 'lad_NL',
            'description': 'Local authority districts for the Netherlands',
            'filename': 'lad_NL_V2.csv'
        }
        config_handler.update_region_definition('lad_NL', region_definition)
        region_definitions = config_handler.read_region_definitions()
        assert len(region_definitions) == 2
        for region_definition in region_definitions:
            if region_definition['name'] == 'lad_NL':
                assert region_definition['filename'] == 'lad_NL_V2.csv'

        # region_definition sets / modify unique identifier (name)
        region_definition['name'] = 'name_change'
        config_handler.update_region_definition('lad_NL', region_definition)
        region_definitions = config_handler.read_region_definitions()
        assert len(region_definitions) == 2
        for region_definition in region_definitions:
            if region_definition['name'] == 'name_change':
                assert region_definition['filename'] == 'lad_NL_V2.csv'

    def test_project_interval_definitions(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # interval_definitions / read existing (from fixture)
        interval_definitions = config_handler.read_interval_definitions()
        assert interval_definitions[0]['name'] == 'hourly'
        assert len(interval_definitions) == 2

        # interval_definition sets / add
        interval_definition = {
            'name': 'monthly',
            'description': 'The 12 months of the year',
            'filename': 'monthly.csv'
        }
        config_handler.write_interval_definition(interval_definition)
        interval_definitions = config_handler.read_interval_definitions()
        assert len(interval_definitions) == 3
        for interval_definition in interval_definitions:
            if interval_definition['name'] == 'monthly':
                assert interval_definition['filename'] == 'monthly.csv'

        # interval_definition sets / modify
        interval_definition = {
            'name': 'monthly',
            'description': 'The 12 months of the year',
            'filename': 'monthly_V2.csv'
        }
        config_handler.update_interval_definition(
            interval_definition['name'], interval_definition)
        interval_definitions = config_handler.read_interval_definitions()
        assert len(interval_definitions) == 3
        for interval_definition in interval_definitions:
            if interval_definition['name'] == 'monthly':
                assert interval_definition['filename'] == 'monthly_V2.csv'

        # region_definition sets / modify unique identifier (name)
        interval_definition['name'] = 'name_change'
        config_handler.update_interval_definition(
            'monthly', interval_definition)
        interval_definitions = config_handler.read_interval_definitions()
        assert len(interval_definitions) == 3
        for interval_definition in interval_definitions:
            if interval_definition['name'] == 'name_change':
                assert interval_definition['filename'] == 'monthly_V2.csv'

    def test_project_scenario_sets(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # Scenario sets / read existing (from fixture)
        scenario_sets = config_handler.read_scenario_sets()
        assert scenario_sets[0]['name'] == 'population'
        assert len(scenario_sets) == 1

        # Scenario sets / add
        scenario_set = {
            'description': 'The annual mortality rate in UK population',
            'name': 'mortality'
        }
        config_handler.write_scenario_set(scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'mortality':
                expected = 'The annual mortality rate in UK population'
                assert scenario_set['description'] == expected

        # Scenario sets / modify
        scenario_set = {
            'description': 'The annual mortality rate in NL population',
            'name': 'mortality'
        }
        config_handler.update_scenario_set(scenario_set['name'], scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'mortality':
                expected = 'The annual mortality rate in NL population'
                assert scenario_set['description'] == expected

        # Scenario sets / modify unique identifier (name)
        scenario_set['name'] = 'name_change'
        config_handler.update_scenario_set('mortality', scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'name_change':
                expected = 'The annual mortality rate in NL population'
                assert scenario_set['description'] == expected

    def test_project_scenarios(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # Scenarios / read existing (from fixture)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 2

        # Scenarios / add
        scenario = {
            'description': 'The Medium ONS Forecast for UK population out to 2050',
            'filename': 'population_medium.csv',
            'name': 'Medium Population (ONS)',
            'parameters': [
                {
                    'name': 'population_count',
                    'spatial_resolution': 'lad',
                    'temporal_resolution': 'annual',
                    'units': 'people',
                }
            ],
            'scenario_set': 'population',
        }
        config_handler.write_scenario(scenario)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'Medium Population (ONS)':
                assert scenario['filename'] == 'population_medium.csv'

        # Scenarios / modify
        scenario['filename'] = 'population_med.csv'
        config_handler.update_scenario(scenario['name'], scenario)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'Medium Population (ONS)':
                assert scenario['filename'] == 'population_med.csv'

        # Scenarios / modify unique identifier (name)
        scenario['name'] = 'name_change'
        config_handler.update_scenario('Medium Population (ONS)', scenario)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'name_change':
                assert scenario['filename'] == 'population_med.csv'


    def test_project_narrative_sets(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # narrative sets / read existing (from fixture)
        narrative_sets = config_handler.read_narrative_sets()
        assert narrative_sets[0]['name'] == 'technology'
        assert len(narrative_sets) == 2

        # narrative sets / add
        narrative_set = {
            'description': 'The rate of development in the UK',
            'name': 'development'
        }
        config_handler.write_narrative_set(narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'development':
                expected = 'The rate of development in the UK'
                assert narrative_set['description'] == expected

        # narrative sets / modify
        narrative_set = {
            'description': 'The rate of technical development in the NL',
            'name': 'technology'
        }
        config_handler.update_narrative_set(narrative_set['name'], narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'technology':
                expected = 'The rate of technical development in the NL'
                assert narrative_set['description'] == expected

        # narrative sets / modify unique identifier (name)
        narrative_set['name'] = 'name_change'
        config_handler.update_narrative_set('technology', narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'name_change':
                expected = 'The rate of technical development in the NL'
                assert narrative_set['description'] == expected

    def test_project_narratives(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # narratives / read existing (from fixture)
        narratives = config_handler.read_narratives()
        assert len(narratives) == 2

        # narratives / add
        narrative = {
            'description': 'The Medium ONS Forecast for UK population out to 2050',
            'filename': 'population_medium.csv',
            'name': 'Medium Population (ONS)',
            'parameters': [
                {
                    'name': 'population_count',
                    'spatial_resolution': 'lad',
                    'temporal_resolution': 'annual',
                    'units': 'people',
                }
            ],
            'narrative_set': 'population',
        }
        config_handler.write_narrative(narrative)
        narratives = config_handler.read_narratives()
        assert len(narratives) == 3
        for narrative in narratives:
            if narrative['name'] == 'Medium Population (ONS)':
                assert narrative['filename'] == 'population_medium.csv'

        # narratives / modify
        narrative['filename'] = 'population_med.csv'
        config_handler.update_narrative(narrative['name'], narrative)
        narratives = config_handler.read_narratives()
        assert len(narratives) == 3
        for narrative in narratives:
            if narrative['name'] == 'Medium Population (ONS)':
                assert narrative['filename'] == 'population_med.csv'

        # narratives / modify unique identifier (name)
        narrative['name'] = 'name_change'
        config_handler.update_narrative('Medium Population (ONS)', narrative)
        narratives = config_handler.read_narratives()
        assert len(narratives) == 3
        for narrative in narratives:
            if narrative['name'] == 'name_change':
                assert narrative['filename'] == 'population_med.csv'


def test_transform_leaves_empty():
    tree = []
    actual = transform_leaves(tree, replace_e)
    assert id(tree) != id(actual)
    assert actual == []


def test_transform_leaves_non_tree():
    tree = "not a tree"
    actual = transform_leaves(tree, replace_e)
    assert actual == tree


def test_transform_list():
    tree = ['a', 'b', 'c', 'd', 'e']
    defensive_copy = deepcopy(tree)
    actual = transform_leaves(tree, replace_e)
    expected = ['a', 'b', 'c', 'd', 'XXX']
    assert actual == expected
    assert tree == defensive_copy


def test_transform_dict():
    tree = {'a': 'e', 'b': ['c', 'e']}
    defensive_copy = deepcopy(tree)
    actual = transform_leaves(tree, replace_e)
    expected = {'a': 'XXX', 'b': ['c', 'XXX']}
    assert actual == expected
    assert tree == defensive_copy


def test_transform_circular():
    tree = {'b': 'e'}
    tree['a'] = tree
    with raises(ValueError) as ex:
        transform_leaves(tree, replace_e)
    assert 'Circular reference detected' in str(ex)


def replace_e(obj, path):
    if obj == 'e':
        return 'XXX'
    else:
        return obj
