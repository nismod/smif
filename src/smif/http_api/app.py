
import pkg_resources

from flask import Flask
from smif.controller import Scheduler
from smif.data_layer import DatafileInterface
from smif.http_api.register import (register_api_endpoints,
                                    register_error_handlers, register_routes)


def get_data_interface():
    """Return a data_layer.DataInterface
    """
    return DatafileInterface(
        pkg_resources.resource_filename('smif', 'sample_project')
    )


def get_scheduler():
    """Return a controller.Scheduler
    """
    return Scheduler()


def create_app(static_folder='static', template_folder='templates',
               data_interface=get_data_interface(), scheduler=get_scheduler()):
    """Create Flask app object
    """
    app = Flask(
        __name__,
        static_url_path='',
        static_folder=static_folder,
        template_folder=template_folder
    )

    app.config.data_interface = data_interface
    app.config.scheduler = scheduler

    register_routes(app)
    register_api_endpoints(app)
    register_error_handlers(app)

    return app
