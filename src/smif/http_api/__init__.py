"""HTTP API endpoint
"""
import smif
import dateutil.parser

from flask import Flask, render_template, request, jsonify, current_app
from flask.views import MethodView

from smif.data_layer import (
    DataExistsError,
    DataMismatchError,
    DataNotFoundError
)


def create_app(static_folder='static', template_folder='templates', get_data_interface=None):
    """Create Flask app object
    """
    app = Flask(
        __name__,
        static_url_path='',
        static_folder=static_folder,
        template_folder=template_folder
    )
    # Pass get_data_interface method which must return an instance of a class
    # implementing DataInterface. There may be a better way!
    app.config.get_data_interface = get_data_interface
    
    register_routes(app)
    register_api_endpoints(app)
    register_error_handlers(app)

    return app


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
                 key='sos_model_run_name', key_type='string')
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


class SmifAPI(MethodView):
    """Implement operations for Smif
    """
    def get(self, key):
        """Get smif details
        version: GET /api/v1/smif/version
        """
        if key == 'version':
            data = smif.__version__
        else:
            data = {}
            data['version'] = smif.__version__
        return jsonify(data)


class SosModelRunAPI(MethodView):
    """Implement CRUD operations for sos_model_run configuration data
    """
    def get(self, sos_model_run_name):
        """Get sos_model_runs
        all: GET /api/v1/sos_model_runs/
        one: GET /api/vi/sos_model_runs/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.get_data_interface()
        if sos_model_run_name is None:
            data = data_interface.read_sos_model_runs()
            response = jsonify(data)
        else:
            data = data_interface.read_sos_model_run(sos_model_run_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a sos_model_run:
        POST /api/v1/sos_model_runs
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form

        data_interface.write_sos_model_run(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, sos_model_run_name):
        """Update a sos_model_run:
        PUT /api/v1/sos_model_runs
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data_interface.update_sos_model_run(sos_model_run_name, data)
        response = jsonify({})
        return response

    def delete(self, sos_model_run_name):
        """Delete a sos_model_run:
        DELETE /api/v1/sos_model_runs
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_sos_model_run(sos_model_run_name)
        response = jsonify({})
        return response


class SosModelAPI(MethodView):
    """Implement CRUD operations for sos_model configuration data
    """
    def get(self, sos_model_name):
        """Get sos_model
        all: GET /api/v1/sos_model/
        one: GET /api/vi/sos_model/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.get_data_interface()
        if sos_model_name is None:
            data = data_interface.read_sos_models()
            response = jsonify(data)
        else:
            data = data_interface.read_sos_model(sos_model_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a sos_model:
        POST /api/v1/sos_model
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form

        data_interface.write_sos_model(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, sos_model_name):
        """Update a sos_model:
        PUT /api/v1/sos_model
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data_interface.update_sos_model(sos_model_name, data)
        response = jsonify({})
        return response

    def delete(self, sos_model_name):
        """Delete a sos_model:
        DELETE /api/v1/sos_model
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_sos_model(sos_model_name)
        response = jsonify({})
        return response


class SectorModelAPI(MethodView):
    """Implement CRUD operations for sector_model configuration data
    """
    def get(self, sector_model_name):
        """Get sector_models
        all: GET /api/v1/sector_models/
        one: GET /api/vi/sector_models/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.get_data_interface()
        if sector_model_name is None:
            data = data_interface.read_sector_models()
            response = jsonify(data)
        else:
            data = data_interface.read_sector_model(sector_model_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a sector_model:
        POST /api/v1/sector_models
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)

        data_interface.write_sector_model(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, sector_model_name):
        """Update a sector_model:
        PUT /api/v1/sector_models
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_sector_model(sector_model_name, data)
        response = jsonify({})
        return response

    def delete(self, sector_model_name):
        """Delete a sector_model:
        DELETE /api/v1/sector_models
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_sector_model(sector_model_name)
        response = jsonify({})
        return response


class ScenarioSetAPI(MethodView):
    """Implement CRUD operations for scenario_sets configuration data
    """
    def get(self, scenario_set_name):
        """Get scenario_sets
        all: GET /api/v1/scenario_sets/
        one: GET /api/vi/scenario_sets/name
        """
        data_interface = current_app.config.get_data_interface()
        if scenario_set_name is None:
            data = data_interface.read_scenario_sets()
            response = jsonify(data)
        else:
            data = data_interface.read_scenario_set(scenario_set_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a scenario_set:
        POST /api/v1/scenario_sets
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)

        data_interface.write_scenario_set(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, scenario_set_name):
        """Update a scenario_set:
        PUT /api/v1/scenario_sets
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_scenario_set(scenario_set_name, data)
        response = jsonify({})
        return response

    def delete(self, scenario_set_name):
        """Delete a scenario_set:
        DELETE /api/v1/scenario_sets
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_scenario_set(scenario_set_name)
        response = jsonify({})
        return response


class ScenarioAPI(MethodView):
    """Implement CRUD operations for scenarios configuration data
    """
    def get(self, scenario_name):
        """Get scenarios
        all: GET /api/v1/scenarios/
        one: GET /api/vi/scenarios/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.get_data_interface()
        if scenario_name is None:
            data = data_interface.read_scenarios()
            response = jsonify(data)
        else:
            data = data_interface.read_scenario(scenario_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a scenario:
        POST /api/v1/scenarios
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)

        data_interface.write_scenario(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, scenario_name):
        """Update a scenario:
        PUT /api/v1/scenarios
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_scenario(scenario_name, data)
        response = jsonify({})
        return response

    def delete(self, scenario_name):
        """Delete a scenario:
        DELETE /api/v1/scenarios
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_scenario(scenario_name)
        response = jsonify({})
        return response


class NarrativeSetAPI(MethodView):
    """Implement CRUD operations for narrative_sets configuration data
    """
    def get(self, narrative_set_name):
        """Get narrative_sets
        all: GET /api/v1/narrative_sets/
        one: GET /api/vi/narrative_sets/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.get_data_interface()
        if narrative_set_name is None:
            data = data_interface.read_narrative_sets()
            response = jsonify(data)
        else:
            data = data_interface.read_narrative_set(narrative_set_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a narrative_set:
        POST /api/v1/narrative_sets
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)

        data_interface.write_narrative_set(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, narrative_set_name):
        """Update a narrative_set:
        PUT /api/v1/narrative_sets
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_narrative_set(narrative_set_name, data)
        response = jsonify({})
        return response

    def delete(self, narrative_set_name):
        """Delete a narrative_set:
        DELETE /api/v1/narrative_sets
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_narrative_set(narrative_set_name)
        response = jsonify({})
        return response


class NarrativeAPI(MethodView):
    """Implement CRUD operations for narratives configuration data
    """
    def get(self, narrative_name):
        """Get narratives
        all: GET /api/v1/narratives/
        one: GET /api/vi/narratives/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.get_data_interface()
        if narrative_name is None:
            data = data_interface.read_narratives()
            response = jsonify(data)
        else:
            data = data_interface.read_narrative(narrative_name)
            response = jsonify(data)

        return response

    def post(self):
        """Create a narrative:
        POST /api/v1/narratives
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)

        data_interface.write_narrative(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, narrative_name):
        """Update a narrative:
        PUT /api/v1/narratives
        """
        data_interface = current_app.config.get_data_interface()
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_narrative(narrative_name, data)
        response = jsonify({})
        return response

    def delete(self, narrative_name):
        """Delete a narrative:
        DELETE /api/v1/narratives
        """
        data_interface = current_app.config.get_data_interface()
        data_interface.delete_narrative(narrative_name)
        response = jsonify({})
        return response


def register_api(app, view, endpoint, url, key='id', key_type='int'):
    """Register a MethodView as an endpoint with CRUD operations at a URL
    """
    view_func = view.as_view(endpoint)
    app.add_url_rule(url, defaults={key: None},
                     view_func=view_func, methods=['GET'])
    app.add_url_rule(url, view_func=view_func, methods=['POST'])
    app.add_url_rule('%s<%s:%s>' % (url, key_type, key), view_func=view_func,
                     methods=['GET', 'PUT', 'DELETE'])


def check_timestamp(data):
    """Check for timestamp and parse to datetime object
    """
    if 'stamp' in data:
        try:
            data['stamp'] = dateutil.parser.parse(data['stamp'])
        except(ValueError):
            pass
    return data
