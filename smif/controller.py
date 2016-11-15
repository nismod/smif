"""This module coordinates the software components that make up the integration
framework.

Folder structure
----------------

When configuring a system-of-systems model, the folder structure below should
be used.  In this example, there is one sector model, called ``water_supply``::

    /models
    /models/water_supply/
    /models/water_supply/run.py
    /models/water_supply/assets/assets1.yaml
    /config/
    /config/model.yaml
    /config/timesteps.yaml
    /planning/
    /planning/pre-specified.yaml

The ``models`` folder contains one subfolder for each sector model.

"""
import logging
import os
from subprocess import check_call

from smif.parse_config import ConfigParser
from smif.runner import ModelRunner

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class Controller:
    """Coordinates the data-layer, decision-layer and model-runner
    """
    def __init__(self, project_folder):
        """
        Arguments
        =========
        project_folder : str
            File path to the project folder

        """
        self._project_folder = project_folder
        self._model_list = []
        self._timesteps = []

        logger.info("Getting config file from {}".format(project_folder))
        self._configuration = self._get_config()

    def _get_config(self):
        """Loads the model configuration from the configuration file

        """
        config_path = os.path.join(self._project_folder,
                                   'config',
                                   'model.yaml')
        msg = "Looking for configuration data in {}".format(config_path)
        logger.info(msg)

        config_parser = ConfigParser(config_path)
        config_parser.validate_as_modelrun_config()

        config_data = config_parser.data

        self.load_models(config_data['sector_models'])
        self._timesteps = self.load_timesteps(config_data['timesteps'])
        planning_config = config_data['planning']
        logger.info("Loading planning config: {}".format(planning_config))
        return config_data

    def load_timesteps(self, file_path):
        """Load the timesteps from the configuration file

        Arguments
        =========
        file_path: str
            The path to the timesteps file

        Returns
        =======
        list
            A list of timesteps
        """
        file_path = os.path.join(self._project_folder,
                                 'config',
                                 str(file_path))
        logger.info("Loading timesteps from {}".format(file_path))
        return ConfigParser(file_path).data

    def load_models(self, model_list):
        """Loads the sector models into the controller

        Arguments
        =========
        model_list : list
            A list of sector model names

        """
        for model in model_list:
            self.load_model(model)

    def load_model(self, model_name):
        """Loads the sector model

        Arguments
        =========
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        logger.info("Loading models: {}".format(model_name))
        self._model_list.append(ModelRunner(self._project_folder, model_name))

    def run_sector_model(self, model_name):
        """Runs the sector model in a subprocess

        Arguments
        =========
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        msg = "Running the {} sector model".format(model_name)
        logger.info(msg)

        model_path = os.path.join(self._project_folder,
                                  'models',
                                  model_name,
                                  'run.py')
        if os.path.exists(model_path):
            # Run up a subprocess to run the simulation
            check_call(['python', model_path])
        else:
            msg = "Cannot find `run.py` for the {} model".format(model_name)
            raise Exception(msg)

    def run_sos_model(self):
        """Runs the system-of-system model
        """
        msg = "Can't run the SOS model yet"
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def timesteps(self):
        """Returns the list of timesteps

        Returns
        =======
        list
            A list of timesteps
        """
        return self._timesteps

    @property
    def all_assets(self):
        """Returns the list of all assets across the system-of-systems model

        Returns
        =======
        list
            A list of all assets across the system-of-systems model
        """
        assets = []
        for model in self._model_list:
            assets.extend(model.assets)
        return assets

    @property
    def model_list(self):
        """The list of sector model names

        Returns
        =======
        list
            A list of sector model names
        """
        models = []
        for model in self._model_list:
            models.append(model.name)
        return models
