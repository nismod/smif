import logging
import os

from smif.parse_config import ConfigParser

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class Controller:
    """Coordinates the data-layer, decision-layer and model-runner
    """
    def __init__(self, project_folder):
        self._project_folder = project_folder
        self._model_list = []
        self._timesteps = []
        self._assets = []

        logger.info("Getting config from {}".format(project_folder))
        self._configuration = self._get_config()

    def _get_config(self):
        config_path = os.path.join(self._project_folder,
                                   'config',
                                   'model.yaml')
        msg = "Looking for config file in {}".format(config_path)
        logger.info(msg)

        config_data = ConfigParser(config_path).data

        self.load_model(config_data['sector_models'])
        self._timesteps = self.load_timesteps(config_data['timesteps'])
        self._assets = self.load_assets(config_data['assets'])
        planning_config = config_data['planning']
        logger.info("Loading planning config: {}".format(planning_config))
        return config_data

    def load_assets(self, file_path):
        assets = []
        for asset in file_path:
            path_to_assetfile = os.path.join(self._project_folder,
                                             'assets',
                                             asset)
            logger.info("Loading assets from {}".format(path_to_assetfile))
            assets.extend(ConfigParser(path_to_assetfile).data)
        return assets

    def load_timesteps(self, file_path):
        file_path = os.path.join(self._project_folder,
                                 'config',
                                 str(file_path[0]))
        logger.info("Loading timesteps from {}".format(file_path))
        return ConfigParser(file_path).data

    def load_models(self, model_list):
        for model in model_list:
            self.load_model(model)

    def load_model(self, model_name):
        logger.info("Loading models: {}".format(model_name))
        self._model_list.extend(model_name)

    def run_sector_model(self, model_name):
        msg = "Can't run the {} sector model yet".format(model_name)
        logger.error(msg)
        raise NotImplementedError(msg)

    def run_sos_model(self):
        msg = "Can't run the SOS model yet"
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def model_list(self):
        return self._model_list
