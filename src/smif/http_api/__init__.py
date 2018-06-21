"""Provide APP constant for the purposes of manually running the flask app

For example, build the front end::
>>> cd smif/app/
>>> npm run build

Or to rebuild js/css on change::
>>> npm run watch

Then run the server app in debug mode with environment variables
>>> FLASK_APP=smif.http_api.app FLASK_DEBUG=1 flask run

On Windows under some Flask versions, use this workaround for 'SyntaxError:
Non-UTF-8 code starting with '\x90' in file flask.exe'
(see https://github.com/pallets/flask/issues/2543):
        FLASK_APP=smif.http_api FLASK_DEBUG=1 python -m flask run

Or if backend debug mode is not needed, just use the smif CLI::
>>> smif app -d ../sample_project

API commands can be operated without the front end.

GET commands can be performed by going to the requisted url::
>>> firefox http://localhost:5000/api/v1/sos_model_runs

POST/PUT commands on configurations can be performed by using the curl utilities under linux::
>>> curl -d '{
...     "name": "scenario_set",
...     "description": "my description",
...     "facets": []
... }' -H "Content-Type: application/json" -X POST http://localhost:5000/api/v1/scenario_sets/

Actions are triggered by sending an empty POST::
>>> curl -d '{}' http://localhost:5000/api/v1/sos_model_runs/20170918_energy_water/start
"""

# import classes for access like ::
#         from smif.http_api import create_app
import pkg_resources
from smif.http_api.app import create_app

# Define what should be imported as * ::
#         from smif.http_api import *
__all__ = ['create_app']

# export the APP so that this can be started like ::
#         FLASK_APP=smif.http_api.app FLASK_DEBUG=1 flask run
APP = create_app(
    static_folder=pkg_resources.resource_filename('smif', 'app/dist'),
    template_folder=pkg_resources.resource_filename('smif', 'app/dist')
)
