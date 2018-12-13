"""Test YAML config store
"""
from pytest import fixture, raises
from smif.data_layer.file.file_config_store import YamlConfigStore
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError)


@fixture(scope='function')
def config_handler(setup_folder_structure, get_sector_model, sample_scenarios):
    handler = YamlConfigStore(str(setup_folder_structure), validation=False)
    for scenario in sample_scenarios:
        handler.write_scenario(scenario)
    handler.write_model(get_sector_model)
    return handler


class TestModelRun:
    """Model runs should be defined once, hard to overwrite
    """
    def test_model_run_read_all(self, model_run, config_handler):
        """Test to write two model_run configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """

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

    def test_model_run_write_twice(self, model_run, config_handler):
        """Test that writing a model_run should fail (not overwrite).
        """

        model_run1 = model_run
        model_run1['name'] = 'unique'
        config_handler.write_model_run(model_run1)

        with raises(SmifDataExistsError) as ex:
            config_handler.write_model_run(model_run1)
        assert "Model_run 'unique' already exists" in str(ex)

    def test_model_run_read_one(self, model_run, config_handler):
        """Test reading a single model_run.
        """

        model_run1 = model_run
        model_run1['name'] = 'model_run1'
        config_handler.write_model_run(model_run1)

        model_run2 = model_run
        model_run2['name'] = 'model_run2'
        config_handler.write_model_run(model_run2)

        model_run = config_handler.read_model_run('model_run2')
        assert model_run['name'] == 'model_run2'

    def test_model_run_read_missing(self, config_handler):
        """Test that reading a missing model_run fails.
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.read_model_run('missing_name')
        assert "Model_run 'missing_name' not found" in str(ex)

    def test_model_run_update(self, model_run, config_handler):
        """Test updating a model_run description
        """
        model_run = model_run
        model_run['name'] = 'to_update'
        model_run['description'] = 'before'

        config_handler.write_model_run(model_run)

        model_run['description'] = 'after'
        config_handler.update_model_run('to_update', model_run)

        actual = config_handler.read_model_run('to_update')
        assert actual['description'] == 'after'

    def test_model_run_update_mismatch(self, model_run, config_handler):
        """Test that updating a model_run with mismatched name should fail
        """
        model_run = model_run

        model_run['name'] = 'model_run'
        with raises(SmifDataMismatchError) as ex:
            config_handler.update_model_run('model_run2', model_run)
        assert "name 'model_run2' must match 'model_run'" in str(ex)

    def test_model_run_update_missing(self, model_run, config_handler):
        """Test that updating a nonexistent model_run should fail
        """
        model_run = model_run
        model_run['name'] = 'missing_name'

        with raises(SmifDataNotFoundError) as ex:
            config_handler.update_model_run('missing_name', model_run)
        assert "Model_run 'missing_name' not found" in str(ex)

    def test_model_run_delete(self, model_run, config_handler):
        """Test that updating a nonexistent model_run should fail
        """
        model_run = model_run
        model_run['name'] = 'to_delete'

        config_handler.write_model_run(model_run)
        before_delete = config_handler.read_model_runs()
        assert len(before_delete) == 1

        config_handler.delete_model_run('to_delete')
        after_delete = config_handler.read_model_runs()
        assert len(after_delete) == 0

    def test_model_run_delete_missing(self, model_run, config_handler):
        """Test that updating a nonexistent model_run should fail
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.delete_model_run('missing_name')
        assert "Model_run 'missing_name' not found" in str(ex)


class TestSosModel:
    """SoSModel configurations should be accessible and editable.
    """
    def test_sos_model_read_all(self, get_sos_model, config_handler):
        """Test to write two sos_model configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """

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

    def test_sos_model_write_twice(self, get_sos_model, config_handler):
        """Test that writing a sos_model should fail (not overwrite).
        """

        sos_model1 = get_sos_model
        sos_model1['name'] = 'unique'
        config_handler.write_sos_model(sos_model1)

        with raises(SmifDataExistsError) as ex:
            config_handler.write_sos_model(sos_model1)
        assert "Sos_model 'unique' already exists" in str(ex)

    def test_sos_model_read_one(self, get_sos_model, config_handler):
        """Test reading a single sos_model.
        """

        sos_model1 = get_sos_model
        sos_model1['name'] = 'sos_model1'
        config_handler.write_sos_model(sos_model1)

        sos_model2 = get_sos_model
        sos_model2['name'] = 'sos_model2'
        config_handler.write_sos_model(sos_model2)

        sos_model = config_handler.read_sos_model('sos_model2')
        assert sos_model['name'] == 'sos_model2'

    def test_sos_model_read_missing(self, config_handler):
        """Test that reading a missing sos_model fails.
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.read_sos_model('missing_name')
        assert "Sos_model 'missing_name' not found" in str(ex)

    def test_sos_model_update(self, get_sos_model, config_handler):
        """Test updating a sos_model description
        """
        sos_model = get_sos_model
        sos_model['name'] = 'to_update'
        sos_model['description'] = 'before'

        config_handler.write_sos_model(sos_model)

        sos_model['description'] = 'after'
        config_handler.update_sos_model('to_update', sos_model)

        actual = config_handler.read_sos_model('to_update')
        assert actual['description'] == 'after'

    def test_sos_model_update_mismatch(self, get_sos_model, config_handler):
        """Test that updating a sos_model with mismatched name should fail
        """
        sos_model = get_sos_model

        sos_model['name'] = 'sos_model'
        with raises(SmifDataMismatchError) as ex:
            config_handler.update_sos_model('sos_model2', sos_model)
        assert "name 'sos_model2' must match 'sos_model'" in str(ex)

    def test_sos_model_update_missing(self, get_sos_model, config_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        sos_model = get_sos_model
        sos_model['name'] = 'missing_name'

        with raises(SmifDataNotFoundError) as ex:
            config_handler.update_sos_model('missing_name', sos_model)
        assert "Sos_model 'missing_name' not found" in str(ex)

    def test_sos_model_delete(self, get_sos_model, config_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        sos_model = get_sos_model
        sos_model['name'] = 'to_delete'

        config_handler.write_sos_model(sos_model)
        before_delete = config_handler.read_sos_models()
        assert len(before_delete) == 1

        config_handler.delete_sos_model('to_delete')
        after_delete = config_handler.read_sos_models()
        assert len(after_delete) == 0

    def test_sos_model_delete_missing(self, get_sos_model, config_handler):
        """Test that updating a nonexistent sos_model should fail
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.delete_sos_model('missing_name')
        assert "Sos_model 'missing_name' not found" in str(ex)


class TestSectorModel:
    """SectorModel definitions should be accessible - may move towards being less editable
    as config, more defined in code/wrapper.
    """
    def test_sector_model_read_all(self, get_sector_model, config_handler):
        """Test to write two sector_model configurations to Yaml files, then
        read the Yaml files and compare that the result is equal.
        """

        sector_model1 = get_sector_model
        sector_model1['name'] = 'sector_model1'
        config_handler.write_model(sector_model1)

        sector_model2 = get_sector_model
        sector_model2['name'] = 'sector_model2'
        config_handler.write_model(sector_model2)

        sector_models = config_handler.read_models()
        sector_model_names = list(sector_model['name'] for sector_model in sector_models)

        assert 'sector_model1' in sector_model_names
        assert 'sector_model2' in sector_model_names
        assert 'energy_demand' in sector_model_names
        assert len(sector_models) == 3

    def test_sector_model_write_twice(self, get_sector_model, config_handler):
        """Test that writing a sector_model should fail (not overwrite).
        """

        sector_model1 = get_sector_model
        sector_model1['name'] = 'unique'
        config_handler.write_model(sector_model1)

        with raises(SmifDataExistsError) as ex:
            config_handler.write_model(sector_model1)
        assert "Sector_model 'unique' already exists" in str(ex)

    def test_sector_model_read_one(self, get_sector_model, config_handler):
        """Test reading a single sector_model.
        """

        sector_model1 = get_sector_model
        sector_model1['name'] = 'sector_model1'
        config_handler.write_model(sector_model1)

        sector_model2 = get_sector_model
        sector_model2['name'] = 'sector_model2'
        config_handler.write_model(sector_model2)

        sector_model = config_handler.read_model('sector_model2')
        assert sector_model['name'] == 'sector_model2'

    def test_sector_model_read_missing(self, config_handler):
        """Test that reading a missing sector_model fails.
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.read_model('missing_name')
        assert "Sector_model 'missing_name' not found" in str(ex)

    def test_sector_model_update(self, get_sector_model, config_handler):
        """Test updating a sector_model description
        """
        sector_model = get_sector_model
        sector_model['name'] = 'to_update'
        sector_model['description'] = 'before'

        config_handler.write_model(sector_model)

        sector_model['description'] = 'after'
        config_handler.update_model('to_update', sector_model)

        actual = config_handler.read_model('to_update')
        assert actual['description'] == 'after'

    def test_sector_model_update_mismatch(self, get_sector_model, config_handler):
        """Test that updating a sector_model with mismatched name should fail
        """
        sector_model = get_sector_model
        sector_model['name'] = 'sector_model2'
        config_handler.write_model(sector_model)

        sector_model['name'] = 'sector_model'
        with raises(SmifDataMismatchError) as ex:
            config_handler.update_model('sector_model2', sector_model)
        assert "name 'sector_model2' must match 'sector_model'" in str(ex)

    def test_sector_model_update_missing(self, get_sector_model, config_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        sector_model = get_sector_model
        sector_model['name'] = 'missing_name'

        with raises(SmifDataNotFoundError) as ex:
            config_handler.update_model('missing_name', sector_model)
        assert "Sector_model 'missing_name' not found" in str(ex)

    def test_sector_model_delete(self, get_sector_model, config_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        before_delete = config_handler.read_models()
        assert len(before_delete) == 1

        config_handler.delete_model('energy_demand')
        after_delete = config_handler.read_models()
        assert len(after_delete) == 0

    def test_sector_model_delete_missing(self, get_sector_model, config_handler):
        """Test that updating a nonexistent sector_model should fail
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.delete_model('missing_name')
        assert "Sector_model 'missing_name' not found" in str(ex)


class TestScenarios:
    """Scenario data should be readable, metadata is currently editable. May move to make it
    possible to import/edit/write data.
    """
    def test_read_scenario_definition(self, setup_folder_structure, config_handler,
                                      sample_scenarios):
        """Should read a scenario definition
        """
        expected = sample_scenarios[0]
        actual = config_handler.read_scenario(expected['name'])
        assert actual == expected

    def test_read_scenario_missing(self, config_handler):
        """Should raise a SmifDataNotFoundError if scenario not found
        """
        with raises(SmifDataNotFoundError) as ex:
            config_handler.read_scenario('missing')
        assert "Scenario 'missing' not found" in str(ex)
