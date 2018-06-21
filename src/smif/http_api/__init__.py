"""Provide method for creating the smif app

For example, build the front end::
>>> cd smif/app/
>>> npm run build

Or to rebuild js/css on change::
>>> npm run watch

Then run the server app in debug mode with environment variables
>>> FLASK_DEBUG=1 smif -v app -d src/smif/sample_project/

Or if backend debug mode is not needed, just use the smif CLI::
>>> smif app -d ../sample_project

API commands can be operated without the front end

GET commands can be performed by going to the requested url::
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
from smif.http_api.app import create_app

# Define what should be imported as * ::
#         from smif.http_api import *
__all__ = ['create_app']
