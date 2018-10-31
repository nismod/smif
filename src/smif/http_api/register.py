from flask import jsonify, render_template
from smif.exception import (SmifDataExistsError, SmifDataMismatchError,
                            SmifDataNotFoundError)
from smif.http_api.crud import (DimensionAPI, ModelRunAPI, NarrativeAPI,
                                ScenarioAPI, SectorModelAPI, SmifAPI,
                                SosModelAPI)


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
    register_api(app, ModelRunAPI, 'model_run_api', '/api/v1/model_runs/',
                 key='model_run_name', key_type='string',
                 action='action', action_type='string')
    register_api(app, SosModelAPI, 'sos_model_api', '/api/v1/sos_models/',
                 key='sos_model_name', key_type='string')
    register_api(app, SectorModelAPI, 'sector_model_api', '/api/v1/sector_models/',
                 key='sector_model_name', key_type='string')
    register_api(app, ScenarioAPI, 'scenario_api', '/api/v1/scenarios/',
                 key='scenario_name', key_type='string')
    register_api(app, NarrativeAPI, 'narrative_api', '/api/v1/narratives/',
                 key='narrative_name', key_type='string')
    register_api(app, DimensionAPI, 'dimension_api', '/api/v1/dimensions/',
                 key='dimension_name', key_type='string')


def register_error_handlers(app):
    """Handle expected errors
    """
    @app.errorhandler(SmifDataExistsError)
    def handle_exists(error):
        """Return 400 Bad Request if data to be created already exists
        """
        response = jsonify({"message": str(error)})
        response.status_code = 400
        return response

    @app.errorhandler(SmifDataMismatchError)
    def handle_mismatch(error):
        """Return 400 Bad Request if data and id/name are mismatched
        """
        response = jsonify({"message": str(error)})
        response.status_code = 400
        return response

    @app.errorhandler(SmifDataNotFoundError)
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
