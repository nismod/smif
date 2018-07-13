from flask import Flask
from smif.http_api.register import (register_api_endpoints,
                                    register_error_handlers, register_routes)


def create_app(static_folder, template_folder, data_interface, scheduler):
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
