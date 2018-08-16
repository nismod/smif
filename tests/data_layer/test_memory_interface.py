from pytest import fixture, raises
from smif.data_layer import DataExistsError, MemoryInterface


class TestScenarios():
    """Read and write scenario data
    """
    def test_write_scenario_data(self, get_remapped_scenario_data):
        """Write to in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler.write_scenario_data('test_scenario', 'parameter', data, spec, 2010)

        assert handler._scenarios[('test_scenario', 'parameter', spec, 2010)] == data

    def test_read_scenario_data(self, get_remapped_scenario_data):
        """Read from in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler._scenarios[('test_scenario', 'parameter', spec, 2010)] = data

        assert handler.read_scenario_data('test_scenario', 'parameter', spec, 2010) == data


class TestScenarioSets:

    def test_write_scenario_set(self):

        handler = MemoryInterface()

        scenario_set = {
            'description': 'The annual mortality rate in UK population',
            'name': 'mortality'
        }

        handler.write_scenario_set(scenario_set)

        assert handler._scenario_sets['mortality'] == scenario_set

    def test_read_scenario_set(self):

        handler = MemoryInterface()

        scenario_set = {
            'description': 'The annual mortality rate in UK population',
            'name': 'mortality'
        }

        handler._scenario_sets['mortality'] = scenario_set

        assert handler.read_scenario_set('mortality') == scenario_set


class TestSosModel:

    @fixture(scope='function')
    def setup_sos_model(self, get_sos_model):
        handler = MemoryInterface()
        handler._sos_models['energy'] = get_sos_model
        return handler

    def test_write_sos_model(self, get_sos_model):
        handler = MemoryInterface()
        handler.write_sos_model(get_sos_model)

        assert handler._sos_models['energy'] == get_sos_model

    def test_write_existing_sos_model(self, setup_sos_model):

        handler = setup_sos_model
        with raises(DataExistsError):
            handler.write_sos_model({'name': 'energy'})

    def test_read_sos_model(self, get_sos_model):

        handler = MemoryInterface()

        sos_model = get_sos_model
        handler._sos_models['energy'] = sos_model

        assert handler.read_sos_model('energy') == sos_model

    def test_update_sos_model(self, setup_sos_model):

        handler = setup_sos_model

        sos_model = handler.read_sos_model('energy')
        sos_model['sector_models'] = ['energy_demand']

        handler.update_sos_model('energy', sos_model)

        assert handler._sos_models['energy'] == sos_model

    def test_delete_sos_model(self, setup_sos_model):

        handler = setup_sos_model
        handler.delete_sos_model('energy')

        assert 'energy' not in handler._sos_models

    def test_read_sos_models(self, setup_sos_model):
        handler = setup_sos_model
        actual = handler.read_sos_models()
        expected = [handler.read_sos_model('energy')]

        assert actual == expected


class TestSosModelRuns:

    @fixture(scope='function')
    def setup_sos_model_run(self, get_sos_model_run):
        handler = MemoryInterface()
        name = get_sos_model_run['name']
        handler._sos_model_runs[name] = get_sos_model_run
        return handler

    def test_write_sos_model_run(self, get_sos_model_run):
        handler = MemoryInterface()
        handler.write_sos_model_run(get_sos_model_run)

        name = get_sos_model_run['name']
        assert handler._sos_model_runs[name] == get_sos_model_run

    def test_read_sos_model_run(self, setup_sos_model_run):
        handler = setup_sos_model_run
        actual = handler.read_sos_model_runs()
        expected = [handler.read_sos_model_run('unique_model_run_name')]

        assert actual == expected

    def test_update_sos_model_run(self, setup_sos_model_run):
        handler = setup_sos_model_run
        name = 'unique_model_run_name'
        sos_model_run = handler.read_sos_model_run(name)
        sos_model_run['description'] = 'blobby'

        handler.update_sos_model_run(name, sos_model_run)

        assert handler._sos_model_runs[name] == sos_model_run

    def test_delete_sos_model_run(self, setup_sos_model_run):
        handler = setup_sos_model_run
        name = 'unique_model_run_name'
        handler.delete_sos_model_run(name)
        assert name not in handler._sos_model_runs
