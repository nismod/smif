"""Test HTTP API application
"""
import datetime
import json
import os

import dateutil.parser
import pytest
import smif
from smif.data_layer import DataExistsError
from smif.http_api import create_app


@pytest.fixture
def app(request, get_handler):
    """Return an app
    """
    def get_data_interface():
        """Return a DataFileInterface
        """
        return get_handler

    test_app = create_app(
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'http'),
        template_folder=os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'http'),
        get_data_interface=get_data_interface
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

def test_get_smif(client, get_handler):
    """GET smif details
    """
    response = client.get('/api/v1/smif/')
    data = parse_json(response)
    assert data['version'] == smif.__version__

def test_get_smif_version(client, get_handler):
    """GET smif version
    """
    response = client.get('/api/v1/smif/version')
    data = parse_json(response)
    assert data == smif.__version__


def test_get_sos_model_runs(client, get_handler, get_sos_model_run):
    """GET all model runs
    """
    response = client.get('/api/v1/sos_model_runs/')
    data = parse_json(response)
    assert len(data) == 0

    get_handler.write_sos_model_run(get_sos_model_run)
    response = client.get('/api/v1/sos_model_runs/')
    data = parse_json(response)
    assert len(data) == 1
    assert data[0]['name'] == get_sos_model_run['name']


def test_get_sos_model_run(client, get_handler, get_sos_model_run):
    """GET single model run
    """
    name = get_sos_model_run['name']
    get_handler.write_sos_model_run(get_sos_model_run)

    response = client.get('/api/v1/sos_model_runs/{}'.format(name))
    data = parse_json(response)
    assert data == get_sos_model_run


def test_get_sos_model_run_missing(client):
    """GET missing model run
    """
    response = client.get('/api/v1/sos_model_runs/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "sos_model_run 'does_not_exist' not found"


def test_create_sos_model_run(client, get_handler, get_sos_model_run):
    """POST model run
    """
    name = 'test_create_sos_model_run'
    get_sos_model_run['name'] = name
    send = serialise_json(get_sos_model_run)
    response = client.post(
        '/api/v1/sos_model_runs/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_sos_model_run(name)
    assert actual == get_sos_model_run


def test_get_sos_models(client, get_handler, get_sos_model):
    """GET all system-of-systems models
    """
    response = client.get('/api/v1/sos_models/')
    data = parse_json(response)
    assert len(data) == 0

    get_handler.write_sos_model(get_sos_model)
    response = client.get('/api/v1/sos_models/')
    data = parse_json(response)
    assert len(data) == 1
    assert data == [get_sos_model]


def test_get_sos_model(client, get_handler, get_sos_model):
    """GET single system-of-systems model
    """
    name = get_sos_model['name']
    get_handler.write_sos_model(get_sos_model)

    response = client.get('/api/v1/sos_models/{}'.format(name))
    data = parse_json(response)
    assert data == get_sos_model


def test_get_sos_model_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/sos_models/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "sos_model 'does_not_exist' not found"


def test_create_sos_model(client, get_handler, get_sos_model):
    """POST system-of-systems model
    """
    name = 'test_create_sos_model'
    get_sos_model['name'] = name
    send = serialise_json(get_sos_model)
    response = client.post(
        '/api/v1/sos_models/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_sos_model(name)
    assert actual == get_sos_model


def test_get_sector_models(client, get_handler, get_sector_model):
    """GET all model runs
    """
    response = client.get('/api/v1/sector_models/')
    data = parse_json(response)
    assert len(data) == 0

    get_handler.write_sector_model(get_sector_model)
    response = client.get('/api/v1/sector_models/')
    data = parse_json(response)
    assert len(data) == 1
    assert data == [get_sector_model]


def test_get_sector_model(client, get_handler, get_sector_model):
    """GET single model run
    """
    name = get_sector_model['name']
    get_handler.write_sector_model(get_sector_model)

    response = client.get('/api/v1/sector_models/{}'.format(name))
    data = parse_json(response)
    assert data == get_sector_model


def test_get_sector_model_missing(client):
    """GET missing model run
    """
    response = client.get('/api/v1/sector_models/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "sector_model 'does_not_exist' not found"


def test_create_sector_model(client, get_handler, get_sector_model):
    """POST model run
    """
    name = 'test_create_sector_model'
    get_sector_model['name'] = name
    send = serialise_json(get_sector_model)
    response = client.post(
        '/api/v1/sector_models/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_sector_model(name)
    assert actual == get_sector_model


def test_get_scenario_sets(client, get_handler, get_scenario_set):
    """GET all scenario_sets
    """
    response = client.get('/api/v1/scenario_sets/')
    data = parse_json(response)
    assert len(data) == 1

    get_handler.write_scenario_set(get_scenario_set)
    response = client.get('/api/v1/scenario_sets/')
    data = parse_json(response)
    assert len(data) == 2
    assert get_scenario_set in data


def test_get_scenario_set(client, get_handler, get_scenario_set):
    """GET single system-of-systems model
    """
    name = get_scenario_set['name']
    get_handler.write_scenario_set(get_scenario_set)

    response = client.get('/api/v1/scenario_sets/{}'.format(name))
    data = parse_json(response)
    assert data == get_scenario_set


def test_get_scenario_set_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/scenario_sets/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "Scenario set 'does_not_exist' not found"


def test_create_scenario_set(client, get_handler, get_scenario_set):
    """POST system-of-systems model
    """
    name = 'test_create_scenario_set'
    get_scenario_set['name'] = name
    send = serialise_json(get_scenario_set)
    response = client.post(
        '/api/v1/scenario_sets/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_scenario_set(name)
    assert actual == get_scenario_set


def test_get_scenarios(client, get_handler, get_scenario):
    """GET all scenarios
    """
    response = client.get('/api/v1/scenarios/')
    data = parse_json(response)
    assert len(data) == 2

    get_handler.write_scenario(get_scenario)
    response = client.get('/api/v1/scenarios/')
    data = parse_json(response)
    assert len(data) == 3
    assert get_scenario in data


def test_get_scenario(client, get_handler, get_scenario):
    """GET single system-of-systems model
    """
    name = get_scenario['name']
    get_handler.write_scenario(get_scenario)

    response = client.get('/api/v1/scenarios/{}'.format(name))
    data = parse_json(response)
    assert data == get_scenario


def test_get_scenario_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/scenarios/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "scenario 'does_not_exist' not found"


def test_create_scenario(client, get_handler, get_scenario):
    """POST system-of-systems model
    """
    name = 'test_create_scenario'
    get_scenario['name'] = name
    send = serialise_json(get_scenario)
    response = client.post(
        '/api/v1/scenarios/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_scenario(name)
    assert actual == get_scenario


def test_get_narrative_sets(client, get_handler, get_narrative_set):
    """GET all narrative_sets
    """
    response = client.get('/api/v1/narrative_sets/')
    data = parse_json(response)
    assert len(data) == 2

    get_narrative_set['name'] = 'non-clashing'
    get_handler.write_narrative_set(get_narrative_set)
    response = client.get('/api/v1/narrative_sets/')
    data = parse_json(response)
    assert len(data) == 3
    assert get_narrative_set in data


def test_get_narrative_set(client, get_handler, get_narrative_set):
    """GET single system-of-systems model
    """
    get_narrative_set['name'] = 'non-clashing-2'
    name = get_narrative_set['name']
    get_handler.write_narrative_set(get_narrative_set)

    response = client.get('/api/v1/narrative_sets/{}'.format(name))
    data = parse_json(response)
    assert data == get_narrative_set


def test_get_narrative_set_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/narrative_sets/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "narrative_set 'does_not_exist' not found"


def test_create_narrative_set(client, get_handler, get_narrative_set):
    """POST system-of-systems model
    """
    name = 'test_create_narrative_set'
    get_narrative_set['name'] = name
    send = serialise_json(get_narrative_set)
    response = client.post(
        '/api/v1/narrative_sets/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_narrative_set(name)
    assert actual == get_narrative_set


def test_get_narratives(client, get_handler, get_narrative):
    """GET all narratives
    """
    response = client.get('/api/v1/narratives/')
    data = parse_json(response)
    assert len(data) == 2

    narrative = get_narrative.as_dict()
    narrative['name'] = 'non-clashing'
    get_handler.write_narrative(narrative)

    response = client.get('/api/v1/narratives/')
    data = parse_json(response)
    assert len(data) == 3
    assert narrative in data


def test_get_narrative(client, get_handler, get_narrative):
    """GET single system-of-systems model
    """
    narrative = get_narrative.as_dict()
    name = narrative['name']
    try:
        get_handler.write_narrative(narrative)
    except DataExistsError:
        get_handler.delete_narrative(narrative['name'])
        get_handler.write_narrative(narrative)

    response = client.get('/api/v1/narratives/{}'.format(name))
    data = parse_json(response)
    assert data == narrative


def test_get_narrative_missing(client):
    """GET missing system-of-systems model
    """
    response = client.get('/api/v1/narratives/does_not_exist')
    assert response.status_code == 404
    data = parse_json(response)
    assert data['message'] == "narrative 'does_not_exist' not found"


def test_create_narrative(client, get_handler, get_narrative):
    """POST system-of-systems model
    """
    name = 'test_create_narrative'
    narrative = get_narrative.as_dict()
    narrative['name'] = name
    send = serialise_json(narrative)
    response = client.post(
        '/api/v1/narratives/',
        data=send,
        content_type='application/json')
    data = parse_json(response)
    assert response.status_code == 201
    assert data['message'] == 'success'

    actual = get_handler.read_narrative(name)
    assert actual == narrative
