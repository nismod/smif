import logging
import os

import yaml

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class Controller:
    """Coordinates the data-layer, decision-layer and model-runner
    """
    def __init__(self, project_folder):
        self._project_folder = project_folder
        logger.info("Getting config from {}".format(project_folder))
        self.get_config()

    def get_config(self):
        config_path = os.path.join(self._project_folder,
                                   'config',
                                   'model.yaml')
        msg = "Looking for config file in {}".format(config_path)
        logger.info(msg)

        with open(config_path, 'r') as config_file:
            config_data = yaml.load(config_file)
        model_list = config_data['sector_models']
        timestep_file = config_data['timesteps']
        asset_files = config_data['assets']
        planning_config = config_data['planning']

        logger.info("Loading models: {}".format(model_list))
        logger.info("Loading timesteps from {}".format(timestep_file))
        logger.info("Loading assets from {}".format(asset_files))
        logger.info("Loading planning config: {}".format(planning_config))

    def run_sector_model(self, model_name):
        msg = "Can't run the {} sector model yet".format(model_name)
        logger.error(msg)
        raise NotImplementedError(msg)

    def run_sos_model(self):
        msg = "Can't run the SOS model yet"
        logger.error(msg)
        raise NotImplementedError(msg)
