"""Test HTTP API application
"""
import datetime
import json
import os

import dateutil.parser
import pytest
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
    return json.loads(response.data.decode('utf-8'), object_hook=timestamp_parser)


def timestamp_parser(json_dict):
    """Parse 'stamp' to python datetime
    """
    if 'stamp' in json_dict:
        try:
            json_dict['stamp'] = dateutil.parser.parse(json_dict['stamp'])
        except(ValueError):
            pass
    return json_dict


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
    assert data == [get_sos_model_run]


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