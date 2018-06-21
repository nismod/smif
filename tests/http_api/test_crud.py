"""Test HTTP API application
"""
import datetime
import json
import os
from unittest.mock import Mock

import pytest
import smif
from flask import current_app
from smif.data_layer import DataNotFoundError
from smif.http_api import create_app


@pytest.fixture
def mock_scheduler():

    def get_status(arg):
        if arg == 'model_never_started':
            return {
                'status': 'unknown'
            }
        elif arg == 'model_started_and_running':
            return {
                'status': 'running',
            }
        elif arg == 'model_started_and_done':
            return {
                'status': 'done',
            }

    attrs = {
        'get_status.side_effect': get_status
    }
    return Mock(**attrs)


@pytest.fixture
def mock_data_interface(get_sos_model_run, get_sos_model, get_sector_model,
                        get_scenario, get_scenario_set, get_narrative, get_narrative_set):

    def read_sos_model_run(arg):
        _check_exist('sos_model_run', arg)
        return get_sos_model_run

    def read_sos_model(arg):
        _check_exist('sos_model', arg)
        return get_sos_model

    def read_sector_model(arg):
        _check_exist('sector_model', arg)
        return get_sector_model

    def read_scenario(arg):
        _check_exist('scenario', arg)
        return get_scenario

    def read_scenario_set(arg):
        _check_exist('scenario_set', arg)
        return get_scenario_set

    def read_narrative(arg):
        _check_exist('narrative', arg)
        return get_narrative

    def read_narrative_set(arg):
        _check_exist('narrative_set', arg)
        return get_narrative_set

    def _check_exist(config, name):
        if name == 'does_not_exist':
            raise DataNotFoundError("%s '%s' not found" % (config, name))

    attrs = {
        'read_sos_model_runs.side_effect': [[get_sos_model_run]],
        'read_sos_model_run.side_effect':  read_sos_model_run,
        'read_sos_models.side_effect': [[get_sos_model]],
        'read_sos_model.side_effect':  read_sos_model,
        'read_sector_models.side_effect': [[get_sector_model]],
        'read_sector_model.side_effect':  read_sector_model,
        'read_scenarios.side_effect': [[get_scenario]],
        'read_scenario.side_effect':  read_scenario,
        'read_scenario_sets.side_effect': [[get_scenario_set]],
        'read_scenario_set.side_effect':  read_scenario_set,
        'read_narratives.side_effect': [[get_narrative]],
        'read_narrative.side_effect':  read_narrative,
        'read_narrative_sets.side_effect': [[get_narrative_set]],
        'read_narrative_set.side_effect':  read_narrative_set,
    }
    return Mock(**attrs)


@pytest.fixture
def app(request, mock_scheduler, mock_data_interface):

    """Return an app
    """
    test_app = create_app(
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'http'),
        template_folder=os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'http'),
        data_interface=mock_data_interface,
        scheduler=mock_scheduler
    )

    with test_app.app_context():
        yield test_app


@pytest.fixture
def client(request, app):
    """Return an API client
    """
    test_client = app.test_client()

    def teardown():
        pass

    request.addfinalizer(teardown)
    return test_client


def parse_json(response):
    """Parse response data
    """
    return json.loads(response.data.decode('utf-8'))


def serialise_json(data):
    return json.dumps(data, default=timestamp_serialiser)


def timestamp_serialiser(obj):
    """Serialist datetime
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()


def test_hello(client):
    """Start with a welcome message
    """
    response = client.get('/')
    assert "Welcome to smif" in str(response.data)


def test_get_smif(client):
    """GET smif details
    """
    response = client.get('/api/v1/smif/')
    data = parse_json(response)
    assert data['version'] == smif.__version__


def test_get_smif_version(client):
    """GET smif version
    """
    response = client.get('/api/v1/smif/version')
    data = parse_json(response)
    assert data == smif.__version__


def test_get_sos_model_runs(client, get_sos_model_run):
    """GET all model runs
    """
    response = client.get('/api/v1/sos_model_runs/')
    assert current_app.config.data_interface.read_sos_model_runs.call_count == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_sos_model_run]


def test_get_sos_model_run(client, get_sos_model_run):
    """GET single model run
    """
    name = get_sos_model_run['name']
    response = client.get('/api/v1/sos_model_runs/{}'.format(name))
    current_app.config.data_interface.read_sos_model_run.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_sos_model_run


def test_get_sos_model_run_missing(client):
    """GET missing system-of-systems model run
    """
    response = client.get('/api/v1/sos_model_runs/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "sos_model_run 'does_not_exist' not found"


def test_create_sos_model_run(client, get_sos_model_run):
    """POST model run
    """
    name = 'test_create_sos_model_run'
    get_sos_model_run['name'] = name
    send = serialise_json(get_sos_model_run)
    response = client.post(
        '/api/v1/sos_model_runs/',
        data=send,
        content_type='application/json')
    current_app.config.data_interface.write_sos_model_run.assert_called_with(get_sos_model_run)

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_start_sos_model_run(client):
    """POST model run START
    """
    # Start a sos_model_run
    response = client.post(
        '/api/v1/sos_model_runs/20170918_energy_water/start',
        data={},
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_get_modelrun_status_modelrun_never_started(client):
    """GET model run STATUS
    """
    # Check if the modelrun is running
    response = client.get(
        '/api/v1/sos_model_runs/model_never_started/status'
    )
    data = parse_json(response)
    assert response.status_code == 200
    assert data['status'] == 'unknown'


def test_get_modelrun_status_modelrun_running(client):
    """GET model run STATUS
    """
    # Check if the modelrun is running
    response = client.get(
        '/api/v1/sos_model_runs/model_started_and_running/status'
    )
    data = parse_json(response)
    assert response.status_code == 200
    assert data['status'] == 'running'


def test_get_modelrun_status_modelrun_done(client):
    """GET model run STATUS
    """
    # Check if the modelrun was successful
    response = client.get(
        '/api/v1/sos_model_runs/model_started_and_done/status'
    )
    data = parse_json(response)
    assert response.status_code == 200
    assert data['status'] == 'done'


def test_get_sos_models(client, get_sos_model):
    """GET all system-of-systems models
    """
    response = client.get('/api/v1/sos_models/')
    assert current_app.config.data_interface.read_sos_models.called == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_sos_model]


def test_get_sos_model(client, get_sos_model):
    """GET single system-of-systems model
    """
    name = get_sos_model['name']
    response = client.get('/api/v1/sos_models/{}'.format(name))
    current_app.config.data_interface.read_sos_model.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_sos_model


def test_get_sos_model_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/sos_models/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "sos_model 'does_not_exist' not found"


def test_create_sos_model(client, get_sos_model):
    """POST system-of-systems model
    """
    name = 'test_create_sos_model'
    get_sos_model['name'] = name
    send = serialise_json(get_sos_model)
    response = client.post(
        '/api/v1/sos_models/',
        data=send,
        content_type='application/json')
    assert current_app.config.data_interface.write_sos_model.called == 1

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_get_sector_models(client, get_sector_model):
    """GET all model runs
    """
    response = client.get('/api/v1/sector_models/')
    assert current_app.config.data_interface.read_sector_models.called == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_sector_model]


def test_get_sector_model(client, get_sector_model):
    """GET single model run
    """
    name = get_sector_model['name']
    response = client.get('/api/v1/sector_models/{}'.format(name))
    current_app.config.data_interface.read_sector_model.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_sector_model


def test_get_sector_model_missing(client):
    """GET missing model run
    """
    response = client.get('/api/v1/sector_models/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "sector_model 'does_not_exist' not found"


def test_create_sector_model(client, get_sector_model):
    """POST model run
    """
    name = 'test_create_sector_model'
    get_sector_model['name'] = name
    send = serialise_json(get_sector_model)
    response = client.post(
        '/api/v1/sector_models/',
        data=send,
        content_type='application/json')
    current_app.config.data_interface.write_sector_model.assert_called_with(get_sector_model)

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_get_scenario_sets(client, get_scenario_set):
    """GET all scenario_sets
    """
    response = client.get('/api/v1/scenario_sets/')
    assert current_app.config.data_interface.read_scenario_sets.called == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_scenario_set]


def test_get_scenario_set(client, get_scenario_set):
    """GET single system-of-systems model
    """
    name = get_scenario_set['name']
    response = client.get('/api/v1/scenario_sets/{}'.format(name))
    current_app.config.data_interface.read_scenario_set.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_scenario_set


def test_get_scenario_set_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/scenario_sets/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "scenario_set 'does_not_exist' not found"


def test_create_scenario_set(client, get_scenario_set):
    """POST system-of-systems model
    """
    name = 'test_create_scenario_set'
    get_scenario_set['name'] = name
    send = serialise_json(get_scenario_set)
    response = client.post(
        '/api/v1/scenario_sets/',
        data=send,
        content_type='application/json')
    current_app.config.data_interface.write_scenario_set.assert_called_with(get_scenario_set)

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_get_scenarios(client, get_scenario):
    """GET all scenarios
    """
    response = client.get('/api/v1/scenarios/')
    assert current_app.config.data_interface.read_scenarios.called == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_scenario]


def test_get_scenario(client, get_scenario):
    """GET single system-of-systems model
    """
    name = get_scenario['name']
    response = client.get('/api/v1/scenarios/{}'.format(name))
    current_app.config.data_interface.read_scenario.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_scenario


def test_get_scenario_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/scenarios/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "scenario 'does_not_exist' not found"


def test_create_scenario(client, get_scenario):
    """POST system-of-systems model
    """
    name = 'test_create_scenario'
    get_scenario['name'] = name
    send = serialise_json(get_scenario)
    response = client.post(
        '/api/v1/scenarios/',
        data=send,
        content_type='application/json')
    current_app.config.data_interface.write_scenario.assert_called_with(get_scenario)

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_get_narrative_sets(client, get_narrative_set):
    """GET all narrative_sets
    """
    response = client.get('/api/v1/narrative_sets/')
    assert current_app.config.data_interface.read_narrative_sets.called == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_narrative_set]


def test_get_narrative_set(client, get_narrative_set):
    """GET single system-of-systems model
    """
    name = get_narrative_set['name']
    response = client.get('/api/v1/narrative_sets/{}'.format(name))
    current_app.config.data_interface.read_narrative_set.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_narrative_set


def test_get_narrative_set_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/narrative_sets/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "narrative_set 'does_not_exist' not found"


def test_create_narrative_set(client, get_narrative_set):
    """POST system-of-systems model
    """
    name = 'test_create_narrative_set'
    get_narrative_set['name'] = name
    send = serialise_json(get_narrative_set)
    response = client.post(
        '/api/v1/narrative_sets/',
        data=send,
        content_type='application/json')
    current_app.config.data_interface.write_narrative_set.assert_called_with(get_narrative_set)

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'


def test_get_narratives(client, get_narrative):
    """GET all narratives
    """
    response = client.get('/api/v1/narratives/')
    assert current_app.config.data_interface.read_narratives.called == 1

    assert response.status_code == 200
    data = parse_json(response)
    assert data == [get_narrative]


def test_get_narrative(client, get_narrative):
    """GET single system-of-systems model
    """
    name = get_narrative['name']
    response = client.get('/api/v1/narratives/{}'.format(name))
    current_app.config.data_interface.read_narrative.assert_called_with(name)

    assert response.status_code == 200
    data = parse_json(response)
    assert data == get_narrative


def test_get_narrative_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/narratives/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "narrative 'does_not_exist' not found"


def test_create_narrative(client, get_narrative):
    """POST system-of-systems model
    """
    name = 'test_create_narrative'
    get_narrative['name'] = name
    send = serialise_json(get_narrative)
    response = client.post(
        '/api/v1/narratives/',
        data=send,
        content_type='application/json')
    current_app.config.data_interface.write_narrative.assert_called_with(get_narrative)

    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'
