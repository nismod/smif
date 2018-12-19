"""HTTP API endpoint
"""
from collections import defaultdict

import dateutil.parser
import smif
from flask import current_app, jsonify, request
from flask.views import MethodView
from smif.exception import (SmifDataError, SmifDataInputError,
                            SmifDataNotFoundError, SmifException,
                            SmifValidationError)


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

        response = jsonify({
            'data': data,
            'error': {}
        })
        return response


class ModelRunAPI(MethodView):
    """Implement CRUD operations for model_run configuration data
    """
    def get(self, model_run_name=None, action=None):
        """Get model_runs
        all: GET /api/v1/model_runs/
        one: GET /api/vi/model_runs/name
        """
        data_interface = current_app.config.data_interface

        try:
            if action is None:
                if model_run_name is None:

                    model_runs = data_interface.read_model_runs()

                    if 'status' in request.args.keys():
                        # filtered: GET /api/v1/model_runs?status=done
                        data = []
                        for model_run in model_runs:
                            status = current_app.config.scheduler.get_status(model_run['name'])
                            if status['status'] == request.args['status']:
                                data.append(model_run)
                    else:
                        # all: GET /api/v1/model_runs/
                        data = []
                        data = model_runs
                else:
                    # one: GET /api/vi/model_runs/name
                    data = {}
                    data = data_interface.read_model_run(model_run_name)
            elif action == 'status':
                # action: GET /api/vi/model_runs/name/status
                data = {}
                data = current_app.config.scheduler.get_status(model_run_name)

            response = jsonify({
                'data': data,
                'error': {}
            })
        except SmifException as err:
            response = jsonify({
                'data': data,
                'error': parse_exceptions(err)
            })

        return response

    def post(self, model_run_name=None, action=None):
        """
        Create a model_run:
        - POST /api/v1/model_runs

        Perform an operation on a model_run
        - POST /api/v1/model_runs/<model_run_name>/<action>

        Available actions are
        - start: Start the model_run
        - kill: Stop a model_run that is currently running
        - remove: Remove a model_run that is waiting to be executed
        - resume: Warm start a model_run
        """
        data_interface = current_app.config.data_interface

        try:
            if action is None:
                data = request.get_json() or request.form
                data_interface.write_model_run(data)
            elif action == 'start':
                data = request.get_json() or request.form
                args = {
                    'verbosity': data['args']['verbosity'],
                    'warm_start': data['args']['warm_start'],
                    'output_format': data['args']['output_format']
                }
                if hasattr(data_interface, 'model_base_folder'):
                    args['directory'] = data_interface.model_base_folder
                current_app.config.scheduler.add(model_run_name, args)
            elif action == 'kill':
                current_app.config.scheduler.kill(model_run_name)
            elif action == 'remove':
                raise NotImplementedError
            elif action == 'resume':
                raise NotImplementedError
            else:
                raise SyntaxError("ModelRun action '%s' does not exist" % action)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 201
        return response

    def put(self, model_run_name):
        """Update a model_run:
        PUT /api/v1/model_runs
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data_interface.update_model_run(model_run_name, data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 200
        return response

    def delete(self, model_run_name):
        """Delete a model_run:
        DELETE /api/v1/model_runs
        """
        data_interface = current_app.config.data_interface
        data_interface.delete_model_run(model_run_name)
        response = jsonify({})
        return response


class SosModelAPI(MethodView):
    """Implement CRUD operations for sos_model configuration data
    """
    def get(self, sos_model_name):
        """Get sos_model
        all: GET /api/v1/sos_models/
        one: GET /api/vi/sos_models/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.data_interface

        try:
            if sos_model_name is None:
                data = []
                data = data_interface.read_sos_models()
            else:
                data = {}
                data = data_interface.read_sos_model(sos_model_name)

            response = jsonify({
                'data': data,
                'error': {}
            })
        except SmifException as err:
            response = jsonify({
                'data': data,
                'error': parse_exceptions(err)
            })

        return response

    def post(self):
        """Create a sos_model:
        POST /api/v1/sos_models
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data_interface.write_sos_model(data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 201
        return response

    def put(self, sos_model_name):
        """Update a sos_model:
        PUT /api/v1/sos_models
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data_interface.update_sos_model(sos_model_name, data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 200
        return response

    def delete(self, sos_model_name):
        """Delete a sos_model:
        DELETE /api/v1/sos_models
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

        try:
            if sector_model_name is None:
                data = []
                data = data_interface.read_models(skip_coords=True)
            else:
                data = {}
                data = data_interface.read_model(sector_model_name, skip_coords=True)

            response = jsonify({
                'data': data,
                'error': {}
            })
        except SmifException as err:
            response = jsonify({
                'data': data,
                'error': parse_exceptions(err)
            })
        return response

    def post(self):
        """Create a sector_model:
        POST /api/v1/sector_models
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form
        data = check_timestamp(data)

        try:
            data_interface.write_model(data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
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

        try:
            data_interface.update_model(sector_model_name, data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 200
        return response

    def delete(self, sector_model_name):
        """Delete a sector_model:
        DELETE /api/v1/sector_models
        """
        data_interface = current_app.config.data_interface
        data_interface.delete_model(sector_model_name)
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

        try:
            if scenario_name is None:
                data = []
                data = data_interface.read_scenarios(skip_coords=True)
            else:
                data = {}
                data = data_interface.read_scenario(scenario_name, skip_coords=True)

            response = jsonify({
                'data': data,
                'error': {}
            })
        except SmifException as err:
            response = jsonify({
                'data': data,
                'error': parse_exceptions(err)
            })

        return response

    def post(self):
        """Create a scenario:
        POST /api/v1/scenarios
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data = check_timestamp(data)
            data_interface.write_scenario(data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 201
        return response

    def put(self, scenario_name):
        """Update a scenario:
        PUT /api/v1/scenarios
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data = check_timestamp(data)
            data_interface.update_scenario(scenario_name, data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 200
        return response

    def delete(self, scenario_name):
        """Delete a scenario:
        DELETE /api/v1/scenarios
        """
        data_interface = current_app.config.data_interface
        data_interface.delete_scenario(scenario_name)
        response = jsonify({})
        return response


class DimensionAPI(MethodView):
    """Implement CRUD operations for dimensions configuration data
    """
    def get(self, dimension_name):
        """Get dimensions
        all: GET /api/v1/dimensions/
        one: GET /api/vi/dimensions/name
        """
        # return str(current_app.config)
        data_interface = current_app.config.data_interface

        try:
            if dimension_name is None:
                data = []
                data = data_interface.read_dimensions(skip_coords=True)
            else:
                data = {}
                data = data_interface.read_dimension(dimension_name, skip_coords=True)

            response = jsonify({
                'data': data,
                'error': {}
            })
        except SmifException as err:
            response = jsonify({
                'data': data,
                'error': parse_exceptions(err)
            })

        return response

    def post(self):
        """Create a dimension:
        POST /api/v1/dimensions
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data = check_timestamp(data)
            data_interface.write_dimension(data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 201
        return response

    def put(self, dimension_name):
        """Update a dimension:
        PUT /api/v1/dimensions
        """
        data_interface = current_app.config.data_interface
        data = request.get_json() or request.form

        try:
            data = check_timestamp(data)
            data_interface.update_dimension(dimension_name, data)
        except SmifException as err:
            response = jsonify({
                'message': 'failed',
                'data': data,
                'error': parse_exceptions(err)
            })
        else:
            response = jsonify({"message": "success"})

        response.status_code = 200
        return response

    def delete(self, dimension_name):
        """Delete a dimension:
        DELETE /api/v1/dimensions
        """
        data_interface = current_app.config.data_interface
        data_interface.delete_dimension(dimension_name)
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


def parse_exceptions(exception):
    """Parse a group of exceptions so that it can be sent over
    the http-api
    """
    if type(exception) == SmifDataError:
        msg = defaultdict(list)
        for ex in exception.args[0]:
            msg[str(type(ex).__name__)].append(_parse_exception(ex))
    else:
        msg = {}
        msg[str(type(exception).__name__)] = [_parse_exception(exception)]

    return msg


def _parse_exception(ex):
    """Parse a single exception so that it can be sent over
    the http-api
    """
    if type(ex) == SmifValidationError:
        msg = ex.args[0]
    if type(ex) == SmifDataInputError:
        msg = {
            'component': ex.component,
            'error': ex.error,
            'message': ex.message,
        }
    if type(ex) == SmifDataNotFoundError:
        msg = ex.args[0]
    return msg
