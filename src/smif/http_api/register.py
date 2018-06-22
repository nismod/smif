from flask import jsonify, render_template
from smif.data_layer import (DataExistsError, DataMismatchError,
                             DataNotFoundError)
from smif.http_api.crud import (NarrativeAPI, NarrativeSetAPI, ScenarioAPI,
                                ScenarioSetAPI, SectorModelAPI, SmifAPI,
                                SosModelAPI, SosModelRunAPI)


def register_routes(app):
    """Register plain routing
    """
    @app.route('/')
    @app.route('/configure')
    @app.route('/configure/<path:path>')
    def home(path=None):
        """Render single page
        """
        return render_template('index.html')


def register_api_endpoints(app):
    """Register API calls (using pluggable views)
    """
    register_api(app, SmifAPI, 'smif_api', '/api/v1/smif/',
                 key='key', key_type='string')
    register_api(app, SosModelRunAPI, 'sos_model_run_api', '/api/v1/sos_model_runs/',
                 key='sos_model_run_name', key_type='string',
                 action='action', action_type='string')
    register_api(app, SosModelAPI, 'sos_model_api', '/api/v1/sos_models/',
                 key='sos_model_name', key_type='string')
    register_api(app, SectorModelAPI, 'sector_model_api', '/api/v1/sector_models/',
                 key='sector_model_name', key_type='string')
    register_api(app, ScenarioSetAPI, 'scenario_set_api', '/api/v1/scenario_sets/',
                 key='scenario_set_name', key_type='string')
    register_api(app, ScenarioAPI, 'scenario_api', '/api/v1/scenarios/',
                 key='scenario_name', key_type='string')
    register_api(app, NarrativeSetAPI, 'narrative_set_api', '/api/v1/narrative_sets/',
                 key='narrative_set_name', key_type='string')
    register_api(app, NarrativeAPI, 'narrative_api', '/api/v1/narratives/',
                 key='narrative_name', key_type='string')


def register_error_handlers(app):
    """Handle expected errors
    """
    @app.errorhandler(DataExistsError)
    def handle_exists(error):
        """Return 400 Bad Request if data to be created already exists
        """
        response = jsonify({"message": str(error)})
        response.status_code = 400
        return response

    @app.errorhandler(DataMismatchError)
    def handle_mismatch(error):
        """Return 400 Bad Request if data and id/name are mismatched
        """
        response = jsonify({"message": str(error)})
        response.status_code = 400
        return response

    @app.errorhandler(DataNotFoundError)
    def handle_not_found(error):
        """Return 404 if data is not found
        """
        response = jsonify({"message": str(error)})
        response.status_code = 404
        return response


def register_api(app, view, endpoint, url, key='id', key_type='int',
                 action=None, action_type=None):
    """Register a MethodView as an endpoint with CRUD operations at a URL
    """
    view_func = view.as_view(endpoint)
    app.add_url_rule(url, defaults={key: None},
                     view_func=view_func, methods=['GET'])
    app.add_url_rule(url, view_func=view_func, methods=['POST'])
    if action:
        app.add_url_rule('%s<%s:%s>/<%s:%s>' % (url, key_type, key, action_type, action),
                         view_func=view_func, methods=['GET'])
        app.add_url_rule('%s<%s:%s>/<%s:%s>' % (url, key_type, key, action_type, action),
                         view_func=view_func, methods=['POST'])
    app.add_url_rule('%s<%s:%s>' % (url, key_type, key), view_func=view_func,
                     methods=['GET', 'PUT', 'DELETE'])
