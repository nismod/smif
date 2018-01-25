"""Provide APP constant for the purposes of manually running the flask app

For example, build the front end, then run the app with environment variables::
        cd smif/app/
        npm run build
        cd ../http_api/
        FLASK_APP=smif.http_api.app FLASK_DEBUG=1 flask run
"""
import os

from smif.data_layer import DatafileInterface
from smif.http_api import create_app


def get_data_interface():
    """Return a data_layer.DataInterface
    """
    return DatafileInterface(
        os.path.join(os.path.dirname(__file__), '..', 'sample_project')
    )


APP = create_app(
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'dist'),
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'dist'),
    get_data_interface=get_data_interface
)
