"""Test data file interface
"""
import csv
import os
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
from pytest import fixture, mark, raises
from smif.data_layer.datafile_interface import DatafileInterface
from smif.data_layer.load import dump
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError)
from smif.metadata import Spec


class TestReadState:

    def test_read_state(self, get_handler):
        handler = get_handler

        modelrun_name = 'modelrun'
        timestep = 2010
        decision_iteration = 0
        dir_ = os.path.join(handler.file_dir['results'], modelrun_name)
        path = os.path.join(dir_, 'state_2010_decision_0.csv')
        os.makedirs(dir_, exist_ok=True)
        with open(path, 'w') as state_fh:
            state_fh.write("build_year,name\n2010,power_station")

        actual = handler.read_state(modelrun_name, timestep, decision_iteration)
        expected = [{'build_year': 2010, 'name': 'power_station'}]
        assert actual == expected

    def test_get_state_filename_all(self, get_handler):

        handler = get_handler

        modelrun_name = 'a modelrun'
        timestep = 2010
        decision_iteration = 0

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)

        expected = os.path.join(
                handler.file_dir['results'], modelrun_name,
                'state_2010_decision_0.csv')

        assert actual == expected

    def test_get_state_filename_none_iteration(self, get_handler):
        handler = get_handler
        modelrun_name = 'a modelrun'
        timestep = 2010
        decision_iteration = None

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)
        expected = os.path.join(handler.file_dir['results'], modelrun_name, 'state_2010.csv')

        assert actual == expected

    def test_get_state_filename_both_none(self, get_handler):
        handler = get_handler
        modelrun_name = 'a modelrun'
        timestep = None
        decision_iteration = None

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)
        expected = os.path.join(
            handler.file_dir['results'], modelrun_name, 'state_0000.csv')

        assert actual == expected

    def test_get_state_filename_timestep_none(self, get_handler):
        handler = get_handler

        modelrun_name = 'a modelrun'
        timestep = None
        decision_iteration = 0

        actual = handler._get_state_filename(modelrun_name, timestep, decision_iteration)
        expected = os.path.join(
            handler.file_dir['results'], modelrun_name, 'state_0000_decision_0.csv')

        assert actual == expected


class TestModelRun:
    """Model runs should be defined once, hard to overwrite
    """
    def test_model_run_read_all(self, model_run, get_handler):
        """Test to write two model_run configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """
        config_handler = get_handler

        model_run1 = model_run
        model_run1['name'] = 'model_run1'
        config_handler.write_model_run(model_run1)

        model_run2 = model_run
        model_run2['name'] = 'model_run2'
        config_handler.write_model_run(model_run2)

        model_runs = config_handler.read_model_runs()
        model_run_names = list(model_run['name'] for model_run in model_runs)

        assert 'model_run1' in model_run_names
        assert 'model_run2' in model_run_names
        assert len(model_runs) == 2

    def test_model_run_write_twice(self, model_run, get_handler):
        """Test that writing a model_run should fail (not overwrite).
        """
        config_handler = get_handler

        model_run1 = model_run
        model_run1['name'] = 'unique'
        config_handler.write_model_run(model_run1)

        with raises(SmifDataExistsError) as ex:
            config_handler.write_model_run(model_run1)
        assert "model_run 'unique' already exists" in str(ex)

    def test_model_run_read_one(self, model_run, get_handler):
        """Test reading a single model_run.
        """
        config_handler = get_handler

        model_run1 = model_run
        model_run1['name'] = 'model_run1'
        config_handler.write_model_run(model_run1)

        model_run2 = model_run
        model_run2['name'] = 'model_run2'
        config_handler.write_model_run(model_run2)

        model_run = config_handler.read_model_run('model_run2')
        assert model_run['name'] == 'model_run2'

    def test_model_run_read_missing(self, get_handler):
        """Test that reading a missing model_run fails.
        """
        with raises(SmifDataNotFoundError) as ex:
            get_handler.read_model_run('missing_name')
        assert "model_run 'missing_name' not found" in str(ex)

    def test_model_run_update(self, model_run, get_handler):
        """Test updating a model_run description
        """
        config_handler = get_handler
        model_run = model_run
        model_run['name'] = 'to_update'
        model_run['description'] = 'before'

        config_handler.write_model_run(model_run)

        model_run['description'] = 'after'
        config_handler.update_model_run('to_update', model_run)

        actual = config_handler.read_model_run('to_update')
        assert actual['description'] == 'after'

    def test_model_run_update_mismatch(self, model_run, get_handler):
        """Test that updating a model_run with mismatched name should fail
        """
        config_handler = get_handler
        model_run = model_run

        model_run['name'] = 'model_run'
        with raises(SmifDataMismatchError) as ex:
            config_handler.update_model_run('model_run2', model_run)
        assert "name 'model_run2' must match 'model_run'" in str(ex)

    def test_model_run_update_missing(self, model_run, get_handler):
        """Test that updating a nonexistent model_run should fail
        """
        config_handler = get_handler
        model_run = model_run
        model_run['name'] = 'missing_name'

        with raises(SmifDataNotFoundError) as ex:
            config_handler.update_model_run('missing_name', model_run)
        assert "model_run 'missing_name' not found" in str(ex)

    def test_model_run_delete(self, model_run, get_handler):
        """Test that updating a nonexistent model_run should fail
        """
        config_handler = get_handler
        model_run = model_run
        model_run['name'] = 'to_delete'

        config_handler.write_model_run(model_run)
        before_delete = config_handler.read_model_runs()
        assert len(before_delete) == 1

        config_handler.delete_model_run('to_delete')
        after_delete = config_handler.read_model_runs()
        assert len(after_delete) == 0

    def test_model_run_delete_missing(self, model_run, get_handler):
        """Test that updating a nonexistent model_run should fail
        """
        config_handler = get_handler
        with raises(SmifDataNotFoundError) as ex:
            config_handler.delete_model_run('missing_name')
        assert "model_run 'missing_name' not found" in str(ex)


class TestSosModel:
    """SoSModel configurations should be accessible and editable.
    """
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

        with raises(SmifDataExistsError) as ex:
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
        with raises(SmifDataNotFoundError) as ex:
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
        with raises(SmifDataMismatchError) as ex:
            config_handler.update_sos_model('sos_model2', sos_model)
        assert "name 'sos_model2' must match 'sos_model'" in str(ex)

    def test_sos_model_update_missing(self, get_sos_model, get_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        config_handler = get_handler
        sos_model = get_sos_model
        sos_model['name'] = 'missing_name'

        with raises(SmifDataNotFoundError) as ex:
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
        with raises(SmifDataNotFoundError) as ex:
            config_handler.delete_sos_model('missing_name')
        assert "sos_model 'missing_name' not found" in str(ex)


class TestSectorModel:
    """SectorModel definitions should be accessible - may move towards being less editable
    as config, more defined in code/wrapper.
    """
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

        with raises(SmifDataExistsError) as ex:
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
        with raises(SmifDataNotFoundError) as ex:
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
        with raises(SmifDataMismatchError) as ex:
            config_handler.update_sector_model('sector_model2', sector_model)
        assert "name 'sector_model2' must match 'sector_model'" in str(ex)

    def test_sector_model_update_missing(self, get_sector_model, get_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        config_handler = get_handler
        sector_model = get_sector_model
        sector_model['name'] = 'missing_name'

        with raises(SmifDataNotFoundError) as ex:
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
        with raises(SmifDataNotFoundError) as ex:
            config_handler.delete_sector_model('missing_name')
        assert "sector_model 'missing_name' not found" in str(ex)

    # itnerventions should be tested as read from files via read_sector_model
    @mark.xfail
    def test_read_sector_model_interventions(self,
                                             get_sector_model,
                                             get_handler):
        config_handler = get_handler

        sector_model = get_sector_model
        sector_model['name'] = 'sector_model'
        sector_model['interventions'] = ['energy_demand.csv']
        config_handler.write_sector_model(sector_model)

        config_handler.read_interventions = Mock(return_value=[{'name': '_an_intervention'}])

        config_handler._read_sector_model_interventions('sector_model')
        assert config_handler.read_interventions.called_with('energy_demand.csv')

    # interventions should be tested as read from files via read_sector_model
    @mark.xfail
    def test_read_interventions(self, get_handler):
        config_handler = get_handler
        config_handler._read_state_file = Mock(return_value=[])
        config_handler._read_yaml_file = Mock(return_value=[])

        config_handler.read_interventions('filename.csv')
        assert config_handler._read_state_file.called_with('filename.csv')

        config_handler.read_interventions('filename.yml')
        assert config_handler._read_yaml_file.called_with('filename.yml')

    def test_reshape_csv_interventions(self, get_handler):
        handler = get_handler

        data = [{'name': 'test', 'capacity_value': 12, 'capacity_unit': 'GW'}]
        expected = [{'name': 'test', 'capacity': {'value': 12, 'unit': 'GW'}}]

        actual = handler._reshape_csv_interventions(data)
        assert actual == expected

    def test_reshape_csv_interventions_duplicate_field(
            self, get_handler):
        handler = get_handler

        data = [{'name': 'test',
                 'capacity_value': 12,
                 'capacity_unit': 'GW',
                 'capacity': 23}]

        with raises(ValueError):
            handler._reshape_csv_interventions(data)

    def test_reshape_csv_interventions_duplicate_field_inv(
            self, get_handler):
        handler = get_handler

        data = [{'name': 'test',
                 'capacity': 23,
                 'capacity_value': 12,
                 'capacity_unit': 'GW'}]

        with raises(ValueError):
            handler._reshape_csv_interventions(data)

    def test_reshape_csv_interventions_underscore_in_name(self, get_handler):
        handler = get_handler

        data = [{'name': 'test', 'mega_capacity_value': 12, 'mega_capacity_unit': 'GW'}]
        expected = [{'name': 'test', 'mega_capacity': {'value': 12, 'unit': 'GW'}}]

        actual = handler._reshape_csv_interventions(data)
        assert actual == expected


# need to test with spec and new methods
@mark.xfail
class TestScenarios:
    """Scenario data should be readable, metadata is currently editable. May move to make it
    possible to import/edit/write data.
    """
    def test_read_scenario_definition(self, setup_folder_structure, get_handler,
                                      sample_scenarios):
        """Should read a scenario definition
        """
        expected = sample_scenarios[1]
        actual = get_handler.read_scenario_definition(expected['name'])
        assert actual == expected

    def test_missing_scenario_definition(self, setup_folder_structure, get_handler):
        """Should raise a SmifDataNotFoundError if scenario definition not found
        """
        with raises(SmifDataNotFoundError) as ex:
            get_handler.read_scenario_definition('missing')
        assert "Scenario definition 'missing' not found" in str(ex)

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
        expected = np.array([[200.0]])
        actual = config_handler.read_scenario_data(
            'High Population (ONS)',
            'population_count',
            'lad',
            'annual',
            2017)

        np.testing.assert_almost_equal(actual, expected)

    def test_scenario_data_validates(self, setup_folder_structure, get_handler,
                                     get_remapped_scenario_data):
        """ DatafileInterface and DataInterface perform validation of scenario
        data against raw interval and region data.

        As such `len(region_names) * len(interval_names)` is not a valid size
        of scenario data under cases where resolution definitions contain
        remapping/resampling info (i.e. multiple hours in a year/regions mapped
        to one name).

        The set of unique region or interval names can be used instead.
        """
        basefolder = setup_folder_structure
        scenario_data, spec = get_remapped_scenario_data

        keys = scenario_data[0].keys()
        with open(os.path.join(str(basefolder), 'data', 'scenarios',
                               'population_high.csv'), 'w+') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(scenario_data)

        config_handler = get_handler

        expected_data = np.array([[100, 150, 200, 210]], dtype=float)
        actual = config_handler.read_scenario_data(
            'High Population (ONS)',
            'population_count',
            'lad',
            'remap_months',
            2015)

        np.testing.assert_equal(actual, expected_data)

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
        scenario_set = {
            'description': 'The annual mortality rate in NL population',
            'name': 'name_change'
        }
        config_handler.update_scenario_set('mortality', scenario_set)
        scenario_sets = config_handler.read_scenario_sets()
        assert len(scenario_sets) == 2
        for scenario_set in scenario_sets:
            if scenario_set['name'] == 'name_change':
                expected = 'The annual mortality rate in NL population'
                assert scenario_set['description'] == expected

    def test_read_scenario_set_missing(self, get_handler):
        """Should raise a SmifDataNotFoundError if scenario set not found
        """
        with raises(SmifDataNotFoundError) as ex:
            get_handler.read_scenario_set('missing')
        assert "Scenario set 'missing' not found" in str(ex)

    def test_project_scenarios(self, get_handler):
        """ Test to read and write the project configuration
        """
        config_handler = get_handler

        # Scenarios / read existing (from fixture)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 2

        # Scenarios / add
        sample_scenario = {
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
        scenario = sample_scenario.copy()
        config_handler.write_scenario(scenario)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'Medium Population (ONS)':
                assert scenario['filename'] == 'population_medium.csv'

        # Scenarios / modify
        scenario = sample_scenario.copy()
        scenario['filename'] = 'population_med.csv'
        config_handler.update_scenario(scenario['name'], scenario)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'Medium Population (ONS)':
                assert scenario['filename'] == 'population_med.csv'

        # Scenarios / modify unique identifier (name)
        scenario = sample_scenario.copy()
        scenario['name'] = 'name_change'
        scenario['filename'] = 'population_medium_change.csv'
        config_handler.update_scenario('Medium Population (ONS)', scenario)
        scenarios = config_handler.read_scenarios()
        assert len(scenarios) == 3
        for scenario in scenarios:
            if scenario['name'] == 'name_change':
                assert scenario['filename'] == 'population_medium_change.csv'

    def test_read_scenario_set_scenario_definitions(self, get_handler):
        """ Test to read all scenario definitions for a scenario
        """
        config_handler = get_handler
        actual = config_handler.read_scenario_set_scenario_definitions('population')
        expected = [
            {
                'description': 'The High ONS Forecast for UK population out to 2050',
                'name': 'High Population (ONS)',
                'facets': [
                    {
                        'name': 'population_count',
                        'filename': 'population_high.csv',
                        'spatial_resolution': 'lad',
                        'temporal_resolution': 'annual',
                        'units': 'people',
                    }
                ],
                'scenario_set': 'population',
            },
            {
                'description': 'The Low ONS Forecast for UK population out to 2050',
                'name': 'Low Population (ONS)',
                'facets': [
                    {
                        'name': 'population_count',
                        'filename': 'population_low.csv',
                        'spatial_resolution': 'lad',
                        'temporal_resolution': 'annual',
                        'units': 'people',
                    }
                ],
                'scenario_set': 'population',
            }
        ]
        assert actual == expected


# need to test with spec and new methods
@mark.xfail
class TestNarratives:
    """Narratives, parameters and interventions should be readable, metadata is editable. May
    move to clarify the distinctions here, and extend to specify strategies and contraints.
    """
    def test_narrative_data(self, setup_folder_structure, get_handler, narrative_data):
        """ Test to dump a narrative (yml) data-file and then read the file
        using the datafile interface. Finally check the data shows up in the
        returned dictionary.
        """
        basefolder = setup_folder_structure
        narrative_data_path = os.path.join(str(basefolder), 'data', 'narratives',
                                           'central_planning.yml')
        dump(narrative_data, narrative_data_path)

        config_handler = get_handler
        test_narrative = config_handler.read_narrative_data('Central Planning')

        assert test_narrative['energy_demand'] == {'smart_meter_savings': 8}

    def test_narrative_data_missing(self, get_handler):
        """Should raise a SmifDataNotFoundError if narrative has no data
        """
        with raises(SmifDataNotFoundError) as ex:
            get_handler.read_narrative_data('missing')
        assert "Narrative 'missing' has no data defined" in str(ex)

    def test_read_narrative_definition(self, setup_folder_structure, get_handler,
                                       sample_narratives):
        expected = sample_narratives[0]
        actual = get_handler.read_narrative(expected['name'])
        assert actual == expected

    def test_read_narrative_definition_missing(self, get_handler):
        """Should raise a SmifDataNotFoundError if narrative not defined
        """
        with raises(SmifDataNotFoundError) as ex:
            get_handler.read_narrative('missing')
        assert "Narrative 'missing' not found" in str(ex)

    def test_read_interventions(self, setup_folder_structure, water_interventions_abc,
                                get_handler):
        path = os.path.join(str(setup_folder_structure), 'data', 'interventions',
                            'reservoirs.yml')
        dump(water_interventions_abc, path)
        actual = get_handler.read_interventions('reservoirs.yml')
        assert actual == water_interventions_abc

    def test_read_initial_conditions(self, setup_folder_structure, initial_system,
                                     get_handler):
        path = os.path.join(str(setup_folder_structure), 'data', 'initial_conditions',
                            'system.yml')
        dump(initial_system, path)
        actual = get_handler.read_initial_conditions('system.yml')
        assert actual == initial_system

    def test_read_strategies(self, setup_folder_structure,
                             initial_system,
                             get_handler):
        path = os.path.join(str(setup_folder_structure), 'data', 'strategies',
                            'a_strategy.yml')
        dump(initial_system, path)
        actual = get_handler.read_strategies('a_strategy.yml')
        assert actual == initial_system

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
        config_handler.update_narrative_set(narrative_set['name'],
                                            narrative_set)
        narrative_sets = config_handler.read_narrative_sets()
        assert len(narrative_sets) == 3
        for narrative_set in narrative_sets:
            if narrative_set['name'] == 'technology':
                expected = 'The rate of technical development in the NL'
                assert narrative_set['description'] == expected

        # narrative sets / modify unique identifier (name)
        narrative_set = {
            'name': 'name_change',
            'description': 'The rate of technical development in the NL'
        }
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

    def test_read_parameters(self, setup_folder_structure, get_handler,
                             model_run, narrative_data):
        """ Test to read a modelrun's parameters
        """
        model_run = model_run
        get_handler.write_model_run(model_run)
        central_narrative_path = os.path.join(
            str(setup_folder_structure),
            'data',
            'narratives',
            'central_planning.yml'
        )
        dump(narrative_data, central_narrative_path)
        high_tech_narrative_path = os.path.join(
            str(setup_folder_structure),
            'data',
            'narratives',
            'energy_demand_high_tech.yml'
        )
        dump(narrative_data, high_tech_narrative_path)

        expected = {
            'smart_meter_savings': 8
        }
        actual = get_handler.read_parameters('unique_model_run_name',
                                             'energy_demand')
        assert actual == expected


# need to test with spec replacing spatial/temporal resolution
@mark.xfail
class TestResults:
    """Results from intermediate stages of running ModelRuns should be writeable and readable.
    """
    def test_read_results(self, setup_folder_structure, get_handler_csv,
                          get_handler_binary):
        """Results from .csv in a folder structure which encodes metadata
        in filenames and directory structure.

        With no decision/iteration specifiers:
            results/
            <modelrun_name>/
            <model_name>/
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv
        Else:
            results/
            <modelrun_name>/
            <model_name>/
            decision_<id>_modelset_<id>/ or decision_<id>/ or modelset_<id>/
                output_<output_name>_
                timestep_<timestep>_
                regions_<spatial_resolution>_
                intervals_<temporal_resolution>.csv
        """
        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'
        output = 'electricity_demand'
        timestep = 2020
        spatial_resolution = 'lad'
        temporal_resolution = 'annual'

        # 1. case with neither modelset nor decision
        expected = np.array([[[1.0]]])
        csv_contents = "region,interval,value\noxford,1,1.0\n"
        binary_contents = pa.serialize(expected).to_buffer()

        path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model,
            "decision_none_modelset_none",
            "output_{}_timestep_{}".format(
                output,
                timestep
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + '.csv', 'w') as fh:
            fh.write(csv_contents)
        actual = get_handler_csv.read_results(modelrun, model, output,
                                              spatial_resolution,
                                              temporal_resolution, timestep)
        assert actual == expected

        with pa.OSFile(path + '.dat', 'wb') as f:
            f.write(binary_contents)
        actual = get_handler_binary.read_results(modelrun, model, output,
                                                 spatial_resolution,
                                                 temporal_resolution, timestep)
        assert actual == expected

        # 2. case with decision
        decision_iteration = 1
        expected = np.array([[[2.0]]])
        csv_contents = "region,interval,value\noxford,1,2.0\n"
        binary_contents = pa.serialize(expected).to_buffer()

        path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model,
            "decision_{}_modelset_none".format(decision_iteration),
            "output_{}_timestep_{}".format(
                output,
                timestep
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + '.csv', 'w') as fh:
            fh.write(csv_contents)
        actual = get_handler_csv.read_results(modelrun, model, output,
                                              spatial_resolution,
                                              temporal_resolution, timestep,
                                              None, decision_iteration)
        assert actual == expected

        with pa.OSFile(path + '.dat', 'wb') as f:
            f.write(binary_contents)
        actual = get_handler_binary.read_results(modelrun, model, output,
                                                 spatial_resolution,
                                                 temporal_resolution, timestep,
                                                 None, decision_iteration)
        assert actual == expected

        # 3. case with modelset
        modelset_iteration = 1
        expected = np.array([[[3.0]]])
        csv_contents = "region,interval,value\noxford,1,3.0\n"
        binary_contents = pa.serialize(expected).to_buffer()
        path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model,
            "decision_none_modelset_{}".format(modelset_iteration),
            "output_{}_timestep_{}".format(
                output,
                timestep
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + '.csv', 'w') as fh:
            fh.write(csv_contents)
        actual = get_handler_csv.read_results(modelrun, model, output,
                                              spatial_resolution,
                                              temporal_resolution, timestep,
                                              modelset_iteration)
        assert actual == expected

        with pa.OSFile(path + '.dat', 'wb') as f:
            f.write(binary_contents)
        actual = get_handler_binary.read_results(modelrun, model, output,
                                                 spatial_resolution,
                                                 temporal_resolution, timestep,
                                                 modelset_iteration)
        assert actual == expected

        # 4. case with both decision and modelset
        expected = np.array([[[4.0]]])
        csv_contents = "region,interval,value\noxford,1,4.0\n"
        binary_contents = pa.serialize(expected).to_buffer()
        path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model,
            "decision_{}_modelset_{}".format(
                modelset_iteration,
                decision_iteration
            ),
            "output_{}_timestep_{}".format(
                output,
                timestep
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path + '.csv', 'w') as fh:
            fh.write(csv_contents)
        actual = get_handler_csv.read_results(modelrun, model, output,
                                              spatial_resolution,
                                              temporal_resolution, timestep,
                                              modelset_iteration,
                                              decision_iteration)
        assert actual == expected

        with pa.OSFile(path + '.dat', 'wb') as f:
            f.write(binary_contents)
        actual = get_handler_binary.read_results(modelrun, model, output,
                                                 spatial_resolution,
                                                 temporal_resolution, timestep,
                                                 modelset_iteration,
                                                 decision_iteration)
        assert actual == expected


class TestWarmStart:
    """If re-running a ModelRun with warm-start specified explicitly, results should be checked
    for existence and left in place.
    """
    def test_prepare_warm_start(self, setup_folder_structure):
        """ Confirm that the warm start copies previous model results
        and reports the correct next timestep
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = DatafileInterface(str(basefolder), 'local_csv')

        # Create results for a 'previous' modelrun
        previous_results_path = os.path.join(
            str(setup_folder_structure),
            "results", modelrun, model,
            "decision_none_modelset_none"
        )
        os.makedirs(previous_results_path, exist_ok=True)

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2020.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,4.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2025.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,6.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2030.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,8.0\n")

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep == 2030

        # Confirm that previous results (excluding the last timestep) exist
        current_results_path = os.path.join(
            str(setup_folder_structure),
            "results", modelrun, model,
            "decision_none_modelset_none"
        )

        warm_start_results = os.listdir(current_results_path)

        assert 'output_electricity_demand_timestep_2020.csv' in warm_start_results
        assert 'output_electricity_demand_timestep_2025.csv' in warm_start_results
        assert 'output_electricity_demand_timestep_2030.csv' not in warm_start_results

    def test_prepare_warm_start_other_local_storage(self, setup_folder_structure):
        """ Confirm that the warm start does not work when previous
        results were saved using a different local storage type
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = DatafileInterface(str(basefolder), 'local_binary')

        # Create results for a 'previous' modelrun
        previous_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(previous_results_path, exist_ok=True)

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2020.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,4.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2025.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,6.0\n")

        path = os.path.join(
            previous_results_path,
            "output_electricity_demand_timestep_2030.csv")
        with open(path, 'w') as fh:
            fh.write("region,interval,value\noxford,1,8.0\n")

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep is None

    def test_prepare_warm_start_no_previous_results(self, setup_folder_structure):
        """ Confirm that the warm start does not work when no previous
        results were saved
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = DatafileInterface(str(basefolder), 'local_binary')

        # Create results for a 'previous' modelrun
        previous_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(previous_results_path, exist_ok=True)

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep is None

        # Confirm that no results were copied
        current_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(current_results_path, exist_ok=True)
        assert len(os.listdir(current_results_path)) == 0

    def test_prepare_warm_start_no_previous_modelrun(self, setup_folder_structure):
        """ Confirm that the warm start does not work when no previous
        modelrun occured
        """

        modelrun = 'energy_transport_baseline'
        model = 'energy_demand'

        # Setup
        basefolder = setup_folder_structure
        current_interface = DatafileInterface(str(basefolder), 'local_binary')

        # Prepare warm start
        current_timestep = current_interface.prepare_warm_start(modelrun)

        # Confirm that the function reports the correct timestep where the model
        # should continue
        assert current_timestep is None

        # Confirm that no results were copied
        current_results_path = os.path.join(
            str(setup_folder_structure),
            "results",
            modelrun,
            model
        )
        os.makedirs(current_results_path, exist_ok=True)
        assert len(os.listdir(current_results_path)) == 0


class TestCoefficients:
    """Dimension conversion coefficients should be cached to disk and read if available.
    """
    @fixture
    def from_spec(self):
        return Spec(name='from_test_coef', dtype='int')

    @fixture
    def to_spec(self):
        return Spec(name='to_test_coef', dtype='int')

    def test_read_write(self, from_spec, to_spec, get_handler):
        data = np.eye(1000)
        handler = get_handler
        handler.write_coefficients(from_spec, to_spec, data)
        actual = handler.read_coefficients(from_spec, to_spec)
        np.testing.assert_equal(actual, data)

    def test_read_raises(self, from_spec, to_spec, get_handler):
        handler = get_handler
        missing_spec = Spec(name='missing_coef', dtype='int')
        with raises(SmifDataNotFoundError):
            handler.read_coefficients(missing_spec, to_spec)

    def test_write_success_if_folder_missing(self, from_spec, to_spec):
        """Ensure we can write files, even if project directory starts empty
        """
        with TemporaryDirectory() as tmpdirname:
            # start with empty project (no data/coefficients subdirectory)
            handler = DatafileInterface(tmpdirname, 'local_binary')
            data = np.eye(10)
            handler.write_coefficients(from_spec, to_spec, data)
