"""HTTP API endpoint
"""
import dateutil.parser
import smif
from flask import current_app, jsonify, request
from flask.views import MethodView


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
    def get(self, sos_model_run_name=None, action=None):
        """Get sos_model_runs
        all: GET /api/v1/sos_model_runs/
        one: GET /api/vi/sos_model_runs/name
        """
        data_interface = current_app.config.data_interface

        if action is None:
            if sos_model_run_name is None:

                sos_model_runs = data_interface.read_sos_model_runs()

                if 'status' in request.args.keys():
                    # filtered: GET /api/v1/sos_model_runs?status=done
                    data = []
                    for sos_model_run in sos_model_runs:
                        status = current_app.config.scheduler.get_status(sos_model_run['name'])
                        if status['status'] == request.args['status']:
                            data.append(sos_model_run)
                else:
                    # all: GET /api/v1/sos_model_runs/
                    data = sos_model_runs
            else:
                # one: GET /api/vi/sos_model_runs/name
                data = data_interface.read_sos_model_run(sos_model_run_name)
        elif action == 'status':
            # action: GET /api/vi/sos_model_runs/name/status
            data = current_app.config.scheduler.get_status(sos_model_run_name)

        return jsonify(data)

    def post(self, sos_model_run_name=None, action=None):
        """
        Create a sos_model_run:
        - POST /api/v1/sos_model_runs

        Perform an operation on a sos_model_run
        - POST /api/v1/sos_model_runs/<sos_model_run_name>/<action>

        Available actions are
        - start: Start the sos_model_run
        - kill: Stop a sos_model_run that is currently running
        - remove: Remove a sos_model_run that is waiting to be executed
        - resume: Warm start a sos_model_run
        """
        data_interface = current_app.config.data_interface

        if action is None:
            data = request.get_json() or request.form
            data_interface.write_sos_model_run(data)
        elif action == 'start':
            data = request.get_json() or request.form
            args = {
                'directory': data_interface.base_folder,
                'verbosity': data['args']['verbosity'],
                'warm_start': data['args']['warm_start'],
                'output_format': data['args']['output_format']
            }
            current_app.config.scheduler.add(sos_model_run_name, args)
        elif action == 'kill':
            current_app.config.scheduler.kill(sos_model_run_name)
        elif action == 'remove':
            raise NotImplementedError
        elif action == 'resume':
            raise NotImplementedError
        else:
            raise SyntaxError("SosModelRun action '%s' does not exist" % action)

        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, sos_model_run_name):
        """Update a sos_model_run:
        PUT /api/v1/sos_model_runs
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data_interface.update_sos_model_run(sos_model_run_name, data)
        response = jsonify({})
        return response

    def delete(self, sos_model_run_name):
        """Delete a sos_model_run:
        DELETE /api/v1/sos_model_runs
        """
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        data_interface.write_sos_model(data)
        response = jsonify({"message": "success"})
        response.status_code = 201
        return response

    def put(self, sos_model_name):
        """Update a sos_model:
        PUT /api/v1/sos_model
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data_interface.update_sos_model(sos_model_name, data)
        response = jsonify({})
        return response

    def delete(self, sos_model_name):
        """Delete a sos_model:
        DELETE /api/v1/sos_model
        """
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_sector_model(sector_model_name, data)
        response = jsonify({})
        return response

    def delete(self, sector_model_name):
        """Delete a sector_model:
        DELETE /api/v1/sector_models
        """
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_scenario_set(scenario_set_name, data)
        response = jsonify({})
        return response

    def delete(self, scenario_set_name):
        """Delete a scenario_set:
        DELETE /api/v1/scenario_sets
        """
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_scenario(scenario_name, data)
        response = jsonify({})
        return response

    def delete(self, scenario_name):
        """Delete a scenario:
        DELETE /api/v1/scenarios
        """
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_narrative_set(narrative_set_name, data)
        response = jsonify({})
        return response

    def delete(self, narrative_set_name):
        """Delete a narrative_set:
        DELETE /api/v1/narrative_sets
        """
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
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
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data = check_timestamp(data)
        data_interface.update_narrative(narrative_name, data)
        response = jsonify({})
        return response

    def delete(self, narrative_name):
        """Delete a narrative:
        DELETE /api/v1/narratives
        """
        data_interface = current_app.config.data_interface
        data_interface.delete_narrative(narrative_name)
        response = jsonify({})
        return response


def check_timestamp(data):
    """Check for timestamp and parse to datetime object
    """
    if 'stamp' in data:
        try:
            data['stamp'] = dateutil.parser.parse(data['stamp'])
        except(ValueError):
            pass
    return data
