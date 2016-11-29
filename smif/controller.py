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
from glob import glob
from importlib.util import module_from_spec, spec_from_file_location
from smif.parse_config import ConfigParser
from smif.sectormodel import SectorModel, SectorModelMode

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)

WRAPPER_FILE_NAME = 'run.py'


class Controller:
    """Coordinates the data-layer, decision-layer and model-runner

    Controller expects to find a yaml configuration file containing
    - lists of assets in ``models/<model_name>/asset_*.yaml``
    - structure of attributes in ``models/<model_name/<asset_name>.yaml

    Controller expects to find a ``run.py`` file in ``models/<model_name>``.
    ``run.py`` contains a python script which subclasses
    :class:`smif.sectormodel.SectorModel` to wrap the sector model.

    """
    def __init__(self, project_folder):
        """
        Arguments
        =========
        project_folder : str
            File path to the project folder

        """
        self._project_folder = project_folder
        self._model_list = {}
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
        for model_name in model_list:
            self._model_list[model_name] = \
                self.load_model(model_name,
                                self._project_folder)

    def load_model(self, model_name, project_folder):
        """Loads the sector model

        Arguments
        =========
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        logger.info("Loading model: {}".format(model_name))

        builder = SectorModelBuilder(model_name, project_folder)
        builder.load_attributes()
        builder.load_wrapper()
        model = builder.finish()
        return model

    def determine_running_mode(self):
        """Determines from the config in what model to run the model

        Returns
        =======
        :class:`SectorModelMode`
            The mode in which to run the model
        """

        number_of_timesteps = len(self.timesteps)

        if number_of_timesteps > 1:
            # Run a sequential simulation
            mode_getter = SectorModelMode()
            mode = mode_getter.get_mode('sequential_simulation')

        elif number_of_timesteps == 0:
            raise ValueError("No timesteps have been specified")

        else:
            # Run a single simulation
            mode_getter = SectorModelMode()
            mode = mode_getter.get_mode('static_simulation')

        return mode

    def run_sector_model(self, model_name):
        """Runs the sector model in a subprocess

        Arguments
        =========
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        msg = "Model {} does not exist. Choose from {}".format(model_name,
                                                               self.model_list)
        assert model_name in self.model_list, msg

        msg = "Running the {} sector model".format(model_name)
        logger.info(msg)

        sector_model = self._model_list[model_name]
        # Run a simulation for a single year (assume no decision vars)
        decision_variables = {}
        sector_model.simulate(decision_variables)

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
        for model in self._model_list.values():
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
        return list(self._model_list.keys())


class SectorConfigReader(object):
    """Parses the models/<sector_model> folder for a configuration files
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.elements = self.get_all_yaml_files()

    def get_all_yaml_files(self):
        pass


class SectorModelBuilder(object):
    """Build the components that make up a sectormodel from the configuration

    """

    def __init__(self, model_name, project_folder):
        self.model_name = model_name
        self._sectormodel = SectorModel(model_name)
        self.project_folder = project_folder

    def load_attributes(self):
        assets = self._load_model_assets()
        attributes = {}
        for asset in assets:
            attributes[asset] = self._load_asset_attributes(asset)
        self._sectormodel.attributes = attributes

    def load_wrapper(self):
        model_path = os.path.join(self.project_folder,
                                  'models',
                                  self.model_name,
                                  WRAPPER_FILE_NAME)
        if os.path.exists(model_path):
            logger.info("Importing run module from {}".format(self.model_name))

            module_path = '{}.run'.format(self.model_name)
            module_spec = spec_from_file_location(module_path, model_path)
            module = module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            self._sectormodel.model = module.wrapper
        else:
            msg = "Cannot find {} for the {} model".format(WRAPPER_FILE_NAME,
                                                           self.model_name)
            raise Exception(msg)

    def validate(self):
        """
        """
        assert self._sectormodel.attributes
        assert self._sectormodel.model

    def finish(self):
        self.validate()
        return self._sectormodel

    def _load_model_assets(self):
        """Loads the assets from the sector model folders

        Using the list of model folders extracted from the configuration file,
        this function returns a list of all the assets from the sector models

        Returns
        =======
        list
            A list of assets from all the sector models

        """
        path_to_assetfile = os.path.join(self.project_folder,
                                         'models',
                                         self.model_name,
                                         'assets',
                                         'asset*')

        for assetfile in glob(path_to_assetfile):
            asset_path = os.path.join(path_to_assetfile, assetfile)
            logger.info("Loading assets from {}".format(asset_path))

        return ConfigParser(asset_path).data

    def _load_asset_attributes(self, asset_name):
        """Loads an asset's attributes into a container

        Arguments
        =========
        asset_name : str
            The name of the asset for which to load attributes

        Returns
        =======
        dict
            A dictionary loaded from the attribute configuration file
        """
        project_folder = self.project_folder
        attribute_path = os.path.join(project_folder, 'models',
                                      self.model_name, 'assets',
                                      "{}.yaml".format(asset_name))
        attributes = ConfigParser(attribute_path).data
        return attributes
