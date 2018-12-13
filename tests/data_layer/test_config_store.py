"""Test all ConfigStore implementations
"""
import getpass
import os
from copy import copy

import smif.data_layer.database.setup_database
from pytest import fixture, mark, param, raises
from smif.data_layer.database_interface import DbConfigStore
from smif.data_layer.file.file_config_store import YamlConfigStore
from smif.data_layer.memory_interface import MemoryConfigStore
from smif.exception import SmifDataExistsError, SmifDataNotFoundError

SKIP_ON_APPVEYOR = mark.skipif(
    'APPVEYOR' in os.environ and os.environ['APPVEYOR'],
    reason="Not yet set up with postgresql service on Appveyor CI")


# For testing, always use `test_smif` database; other connection details are configurable
DB_OPTIONS = {
    'dbname': 'test_smif'
}

# Try setting sane defaults if environment variables are not set
try:
    DB_OPTIONS['host'] = os.environ['PGHOST']
except KeyError:
    DB_OPTIONS['host'] = 'localhost'

try:
    DB_OPTIONS['port'] = os.environ['PGPORT']
except KeyError:
    DB_OPTIONS['port'] = 5432

try:
    DB_OPTIONS['user'] = os.environ['PGUSER']
except KeyError:
    DB_OPTIONS['user'] = getpass.getuser()

try:
    DB_OPTIONS['password'] = os.environ['PGPASSWORD']
except KeyError:
    DB_OPTIONS['password'] = ''


@fixture(
    params=[
        'memory',
        'text_file',
        # param('database', marks=SKIP_ON_APPVEYOR)
        param('database', marks=mark.skip)
    ])
def init_handler(request, setup_empty_folder_structure):
    if request.param == 'memory':
        handler = MemoryConfigStore()
    elif request.param == 'text_file':
        base_folder = setup_empty_folder_structure
        handler = YamlConfigStore(base_folder)
    elif request.param == 'database':
        # migrations dir
        migrations_dir = os.path.join(os.path.dirname(os.path.realpath(
            smif.data_layer.database.setup_database.__file__)), 'migrations')

        # run up migrations
        smif.data_layer.database.setup_database.up_migrations(
            migrations_dir, DB_OPTIONS['dbname'])

        # run down migrations with fixture teardown
        def teardown():
            smif.data_layer.database.setup_database.down_migrations(
                migrations_dir, DB_OPTIONS['dbname'])
        request.addfinalizer(teardown)

        handler = DbConfigStore(**DB_OPTIONS)

    return handler


@SKIP_ON_APPVEYOR
def test_db_connection():
    """Test that we can connect to a database in the test environment
    """
    store = DbConfigStore(**DB_OPTIONS)
    with store.database_connection.cursor() as cur:
        cur.execute('SELECT 1;')
        assert cur.fetchone() == (1,)


@fixture
def handler(init_handler, minimal_model_run, get_sos_model, get_sector_model_no_coords,
            strategies, scenario_no_coords):
    handler = init_handler

    # scenarios
    handler.write_scenario(scenario_no_coords)

    # models
    handler.write_model(get_sector_model_no_coords)

    # sos models
    handler.write_sos_model(get_sos_model)

    # model runs
    handler.write_model_run(minimal_model_run)

    # planning
    handler.write_strategies('test_modelrun', strategies)

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

    def test_read_non_existing_model_run(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_model_run('non_existing')

    def test_write_model_run(self, handler, minimal_model_run):
        new_model_run = {
            'name': 'new_model_run_name',
            'description': 'Model run 2'
        }
        handler.write_model_run(new_model_run)
        actual = handler.read_model_runs()
        expected = [minimal_model_run, new_model_run]
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_write_existing_model_run(self, handler):
        with raises(SmifDataExistsError):
            handler.write_model_run(handler.read_model_run('test_modelrun'))

    def test_update_model_run(self, handler):
        updated_model_run = {
            'name': 'test_modelrun',
            'description': 'Model run'
        }
        handler.update_model_run('test_modelrun', updated_model_run)
        assert handler.read_model_runs() == [updated_model_run]

    def test_update_non_existing_model_run(self, handler):
        with raises(SmifDataNotFoundError):
            handler.update_model_run('non_existing', {'name': 'non_existing'})

    def test_delete_model_run(self, handler):
        handler.delete_model_run('test_modelrun')
        assert handler.read_model_runs() == []

    def test_delete_non_existing_model_run(self, handler):
        with raises(SmifDataNotFoundError):
            handler.delete_model_run('non_existing')


class TestSosModel:
    """Read, write, update, delete SosModel config
    """
    def test_read_sos_models(self, handler, get_sos_model):
        actual = handler.read_sos_models()
        expected = [get_sos_model]
        assert actual == expected

    def test_read_sos_model(self, handler, get_sos_model):
        assert handler.read_sos_model('energy') == get_sos_model

    def test_read_non_existing_sos_model(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_sos_model('non_existing')

    def test_write_sos_model(self, handler, get_sos_model):
        new_sos_model = copy(get_sos_model)
        new_sos_model['name'] = 'another_sos_model'
        handler.write_sos_model(new_sos_model)
        actual = handler.read_sos_models()
        expected = [get_sos_model, new_sos_model]
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_write_existing_sos_model(self, handler):
        with raises(SmifDataExistsError):
            handler.write_sos_model({'name': 'energy'})

    def test_update_sos_model(self, handler, get_sos_model):
        updated_sos_model = copy(get_sos_model)
        updated_sos_model['sector_models'] = ['energy_demand']
        handler.update_sos_model('energy', updated_sos_model)
        assert handler.read_sos_models() == [updated_sos_model]

    def test_update_non_existing_sos_model(self, handler):
        with raises(SmifDataNotFoundError):
            handler.update_sos_model('non_existing', {'name': 'non_existing'})

    def test_delete_sos_model(self, handler):
        handler.delete_sos_model('energy')
        assert handler.read_sos_models() == []

    def test_delete_non_existing_sos_model(self, handler):
        with raises(SmifDataNotFoundError):
            handler.delete_sos_model('non_existing')


class TestSectorModel():
    """Read/write/update/delete SectorModel config
    """
    def test_read_models(self, handler, get_sector_model_no_coords):
        actual = handler.read_models()
        expected = [get_sector_model_no_coords]
        assert actual == expected

    def test_read_model(self, handler, get_sector_model_no_coords):
        actual = handler.read_model(get_sector_model_no_coords['name'])
        expected = get_sector_model_no_coords
        assert actual == expected

    def test_read_non_existing_model(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_model('non_existing')

    def test_write_model(self, handler, get_sector_model_no_coords):
        new_sector_model = copy(get_sector_model_no_coords)
        new_sector_model['name'] = 'another_energy_sector_model'
        handler.write_model(new_sector_model)
        actual = handler.read_models()
        expected = [get_sector_model_no_coords, new_sector_model]
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_write_existing_model(self, handler, get_sector_model_no_coords):
        with raises(SmifDataExistsError):
            handler.write_model(handler.read_model(get_sector_model_no_coords['name']))

    def test_update_model(self, handler, get_sector_model_no_coords):
        name = get_sector_model_no_coords['name']
        expected = copy(get_sector_model_no_coords)
        expected['description'] = ['Updated description']
        handler.update_model(name, expected)
        actual = handler.read_model(name)
        assert actual == expected

    def test_update_non_existing_model(self, handler):
        with raises(SmifDataNotFoundError):
            handler.update_model('non_existing', {'name': 'non_existing'})

    def test_delete_model(self, handler, get_sector_model_no_coords):
        handler.delete_model(get_sector_model_no_coords['name'])
        expected = []
        actual = handler.read_models()
        assert actual == expected

    def test_delete_non_existing_model(self, handler):
        with raises(SmifDataNotFoundError):
            handler.delete_model('non_existing')


class TestStrategies():
    """Read strategies data
    """
    def test_read_strategies(self, handler, strategies):
        expected = strategies
        actual = handler.read_strategies('test_modelrun')
        assert sorted(actual, key=lambda d: d['description']) == \
            sorted(expected, key=lambda d: d['description'])


class TestScenarios():
    """Read and write scenario data
    """
    def test_read_scenarios(self, scenario_no_coords, handler):
        actual = handler.read_scenarios()
        assert actual == [scenario_no_coords]

    def test_read_scenario(self, scenario_no_coords, handler):
        actual = handler.read_scenario('mortality')
        assert actual == scenario_no_coords

    def test_read_non_existing_test_read_scenario(self, handler):
        with raises(SmifDataNotFoundError):
            handler.read_scenario('non_existing')

    def test_write_scenario(self, scenario_no_coords, handler):
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

    def test_write_existing_scenario(self, handler, get_sector_model_no_coords):
        with raises(SmifDataExistsError):
            handler.write_scenario(handler.read_scenario('mortality'))

    def test_update_scenario(self, scenario_no_coords, handler):
        another_scenario = {
            'name': 'mortality',
            'description': 'Projected annual mortality rates',
            'variants': [],
            'provides': []
        }
        handler.update_scenario('mortality', another_scenario)
        assert handler.read_scenarios() == [another_scenario]

    def test_update_non_existing_scenario(self, handler):
        with raises(SmifDataNotFoundError):
            handler.update_scenario('non_existing', {})

    def test_delete_scenario(self, handler):
        handler.delete_scenario('mortality')
        assert handler.read_scenarios() == []

    def test_delete_non_existing_scenario(self, handler):
        with raises(SmifDataNotFoundError):
            handler.delete_scenario('non_existing')

    def test_read_scenario_variants(self, handler, scenario_no_coords):
        actual = handler.read_scenario_variants('mortality')
        expected = scenario_no_coords['variants']
        assert actual == expected

    def test_read_scenario_variant(self, handler, scenario_no_coords):
        actual = handler.read_scenario_variant('mortality', 'low')
        expected = scenario_no_coords['variants'][0]
        assert actual == expected

    def test_write_scenario_variant(self, handler, scenario_no_coords):
        new_variant = {
            'name': 'high',
            'description': 'Mortality (High)',
            'data': {
                'mortality': 'mortality_high.csv'
            }
        }
        handler.write_scenario_variant('mortality', new_variant)
        actual = handler.read_scenario_variants('mortality')
        expected = [new_variant] + scenario_no_coords['variants']
        assert sorted_by_name(actual) == sorted_by_name(expected)

    def test_update_scenario_variant(self, handler):
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

    def test_delete_scenario_variant(self, handler):
        handler.delete_scenario_variant('mortality', 'low')
        assert handler.read_scenario_variants('mortality') == []


def sorted_by_name(list_):
    """Helper to sort lists-of-dicts
    """
    return sorted(list_, key=lambda d: d['name'])
