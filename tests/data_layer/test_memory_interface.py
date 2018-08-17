from pytest import fixture, raises
from smif.data_layer import DataExistsError, MemoryInterface


class TestScenarios():
    """Read and write scenario data
    """
    def test_write_scenario_variant_data(self, get_remapped_scenario_data):
        """Write to in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler.write_scenario_variant_data(
            data, 'test_scenario', 'variant', 'parameter', 2010)
        assert handler._scenario_data[('test_scenario', 'variant', 'parameter', 2010)] == data

    def test_read_scenario_variant_data(self, get_remapped_scenario_data):
        """Read from in-memory data
        """
        data, spec = get_remapped_scenario_data
        handler = MemoryInterface()
        handler._scenario_data[('test_scenario', 'variant', 'parameter', 2010)] = data
        assert handler.read_scenario_variant_data(
            'test_scenario', 'variant', 'parameter', 2010) == data

    def test_write_scenario(self):
        handler = MemoryInterface()
        scenario = {
            'description': 'The annual mortality rate in UK population',
            'name': 'mortality'
        }
        handler.write_scenario(scenario)
        assert handler._scenarios['mortality'] == scenario

    def test_read_scenario(self):
        handler = MemoryInterface()
        scenario = {
            'description': 'The annual mortality rate in UK population',
            'name': 'mortality'
        }
        handler._scenarios['mortality'] = scenario
        assert handler.read_scenario('mortality') == scenario


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


class TestModelRuns:
    """Read, write, update model runs
    """
    def test_write_model_run(self, model_run):
        handler = MemoryInterface()
        handler.write_model_run(model_run)
        assert handler._model_runs[model_run['name']] == model_run

    def test_read_model_run(self, handler):
        actual = handler.read_model_runs()
        expected = [handler.read_model_run('unique_model_run_name')]
        assert actual == expected

    def test_update_model_run(self, handler):
        name = 'unique_model_run_name'
        model_run = handler.read_model_run(name)
        model_run['description'] = 'blobby'
        handler.update_model_run(name, model_run)
        assert handler._model_runs[name] == model_run

    def test_delete_model_run(self, handler):
        name = 'unique_model_run_name'
        handler.delete_model_run(name)
        assert name not in handler._model_runs


@fixture(scope='function')
def model_run():
    return {
        'name': 'unique_model_run_name'
    }


@fixture(scope='function')
def handler(model_run):
    handler = MemoryInterface()
    name = model_run['name']
    handler._model_runs[name] = model_run
    return handler
