"""Provide APP constant for the purposes of manually running the flask app

For example, build the front end::
        cd smif/app/
        npm run build

Or to rebuild js/css on change:
        npm run watch

Then run the server app in debug mode with environment variables
        FLASK_APP=smif.http_api.app FLASK_DEBUG=1 flask run

On Windows under some Flask versions, use this workaround for 'SyntaxError:
Non-UTF-8 code starting with '\x90' in file flask.exe'
(see https://github.com/pallets/flask/issues/2543):
        FLASK_APP=smif.http_api FLASK_DEBUG=1 python -m flask run

Or if backend debug mode is not needed, just use the smif CLI:
        smif app -d ../sample_project
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
