"""This module coordinates the software components that make up the integration
framework.

Folder structure
----------------

When configuring a system-of-systems model, the folder structure below should
be used.  In this example, there is one sector model, called ``water_supply``::

    /models
    /models/water_supply/
    /models/water_supply/run.py
    /models/water_supply/inputs.yaml
    /models/water_supply/outputs.yaml
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

import numpy as np
from smif.decision import Planning
from smif.parse_config import ConfigParser
from smif.sectormodel import SectorModel, SectorModelMode

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)

WRAPPER_FILE_NAME = 'run.py'


class SoSModelReader(object):
    """Encapsulates the parsing of the system-of-systems configuration

    """
    def __init__(self, project_folder):
        logger.info("Getting config file from %s", project_folder)
        self._project_folder = project_folder
        self._configuration = self.parse_sos_config(project_folder)
        self.elements = ['timesteps', 'sector_models', 'planning']
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, builder_instance):
        self._builder = builder_instance

    def parse_sos_config(self, project_folder):
        """
        """
        config_path = os.path.join(project_folder,
                                   'config',
                                   'model.yaml')
        msg = "Looking for configuration data in %s", config_path
        logger.info(msg)

        config_parser = ConfigParser(config_path)
        config_parser.validate_as_modelrun_config()

        return config_parser.data

    def construct(self):
        for element in self.elements:
            if element == 'timesteps':
                timestep_path = self._configuration['timesteps']
                file_path = os.path.join(self._project_folder,
                                         'config',
                                         str(timestep_path))
                self.builder.load_timesteps(file_path)
            elif element == 'sector_models':
                models = self._configuration['sector_models']
                self.builder.load_models(models, self._project_folder)
            elif element == 'planning':
                planning = self._configuration['planning']
                file_paths = [os.path.join(self._project_folder,
                                           'planning',
                                           filename)
                              for filename in
                              planning['pre_specified']['files']]
                self.builder.load_planning(file_paths)


class SosModel(object):
    """Consists of the collection of timesteps and sector models

    Sector models may be joined through dependencies

    This is NISMOD - i.e. the system of system model which brings all of the
    sector models together.
    """
    def __init__(self):
        self.model_list = {}
        self._timesteps = []
        self.planning = None

    @staticmethod
    def run_sos_model():
        """Runs the system-of-system model

        1. Determine running order
        2. Run each sector model
        3. Return success or failure
        """
        msg = "Can't run the SOS model yet"
        logger.error(msg)
        raise NotImplementedError(msg)

    def determine_running_mode(self):
        """Determines from the config in what mode to run the model

        Returns
        =======
        :class:`SectorModelMode`
            The mode in which to run the model
        """

        number_of_timesteps = len(self._timesteps)

        if number_of_timesteps > 1:
            # Run a sequential simulation
            mode = SectorModelMode.sequential_simulation

        elif number_of_timesteps == 0:
            raise ValueError("No timesteps have been specified")

        else:
            # Run a single simulation
            mode = SectorModelMode.static_simulation

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

        sector_model = self.model_list[model_name]
        # Run a simulation for a single year (assume no decision vars)
        decision_variables = np.zeros(2)
        sector_model.simulate(decision_variables)

    @property
    def timesteps(self):
        """Returns the list of timesteps

        Returns
        =======
        list
            A list of timesteps
        """
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._timesteps = value

    @property
    def all_assets(self):
        """Returns the list of all assets across the system-of-systems model

        Returns
        =======
        list
            A list of all assets across the system-of-systems model
        """
        assets = []
        for model in self.model_list.values():
            assets.extend(model.assets)
        return assets

    @property
    def sector_models(self):
        """The list of sector model names

        Returns
        =======
        list
            A list of sector model names
        """
        return list(self.model_list.keys())

    def optimise(self):
        """Runs a dynamic optimisation over a system-of-simulation models

        Use dynamic programming with memoization where the objective function
        :math:`Z(s)` are indexed by state :math:`s`
        """
        pass

    def sequential_simulation(self, model, inputs, decisions):
        results = []
        for index in range(len(self.timesteps)):
            # Intialise the model
            model.inputs = inputs
            # Update the state from the previous year
            if index > 0:
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                logger.debug("Updating %s with %s", state_var, state_res)
                model.inputs.parameters.update_value(state_var, state_res)

            # Run the simulation
            decision = decisions[index]
            results.append(model.simulate(decision))
        return results


class SoSModelBuilder(object):
    """Constructs a system-of-systems model
    """
    def __init__(self):
        self.sos_model = SosModel()

    def load_timesteps(self, file_path):
        """Load the timesteps from the configuration file

        Arguments
        =========
        file_path: str
            The path to the timesteps file
        """
        logger.info("Loading timesteps from %s", file_path)

        cp = ConfigParser(file_path)
        cp.validate_as_timesteps()
        self.set_timesteps(cp.data)

    def set_timesteps(self, timesteps):
        """Set the timesteps of the system-of-systems model

        Arguments
        =========
        list
            A list of timesteps
        """
        self.sos_model.timesteps = timesteps

    def load_planning(self, file_paths):
        """Loads the planning logic into the system of systems model

        Arguments
        =========
        file_paths : list
            A list of file paths

        """
        planning = []
        for filepath in file_paths:
            parser = ConfigParser(filepath)
            parser.validate_as_pre_specified_planning()
            planning.extend(parser.data)
        self.sos_model.planning = Planning(planning)

    def load_models(self, model_list, project_folder):
        """Loads the sector models into the system-of-systems model

        Arguments
        =========
        model_list : list
            A list of sector model names

        """
        for model_name in model_list:
            model = self.load_model(model_name, project_folder)
            self.add_model(model)

    def load_model(self, model_name, project_folder):
        """Loads the sector model into the system-of-systems model

        Arguments
        =========
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        logger.info("Loading model: %s", model_name)

        reader = SectorConfigReader(model_name, project_folder)
        builder = SectorModelBuilder()
        reader.builder = builder

        reader.construct()

        model = builder.finish()
        return model

    def _check_planning_assets_exist(self):
        """Check existence of all the assets in the pre-specifed planning

        """
        model = self.sos_model
        sector_assets = model.all_assets
        for planning_asset in model.planning.assets:
            msg = "Asset '{}' in planning file not found in sector assets"
            assert planning_asset in sector_assets, msg.format(planning_asset)

    def _check_planning_timeperiods_exist(self):
        """Check existence of all the timeperiods in the pre-specified planning
        """
        model = self.sos_model
        model_timeperiods = model.timesteps
        for timeperiod in model.planning.timeperiods:
            msg = "Timeperiod '{}' in planning file not found model config"
            assert timeperiod in model_timeperiods, msg.format(timeperiod)

    def validate(self):
        """Validates the sos model
        """
        self._check_planning_assets_exist()
        self._check_planning_timeperiods_exist()

    def add_model(self, model):
        """Adds a sector model into the system-of-systems model
        """
        self.sos_model.model_list[model.name] = model

    def check_dependencies(self):
        # for each model, compare dependency list of from_models
        # against list of available models
        models_available = self.sos_model.sector_models
        for model_name, model in self.sos_model.model_list.items():
            for dep in model.model.inputs.dependencies:
                if dep.from_model not in models_available:
                    # report missing dependency type
                    raise AssertionError("Missing dependency: {} depends on {} from {}, which is not supplied.".format(model_name, dep.name, dep.from_model))

    def finish(self):
        """Returns a configured system-of-systems model ready for operation

        - includes validation steps, e.g. to check dependencies
        """
        self.validate()
        self.check_dependencies()
        return self.sos_model


class Controller:
    """Coordinates the data-layer, decision-layer and model-runner

    Controller expects to find a yaml configuration file containing
    - lists of assets in ``models/<model_name>/asset_*.yaml``
    - structure of attributes in ``models/<model_name/<asset_name>.yaml``

    Controller expects to find a `WRAPPER_FILE_NAME` file in
    ``models/<model_name>``. `WRAPPER_FILE_NAME` contains a python script
    which subclasses :class:`smif.abstract.AbstractModelWrapper` to wrap the
    sector model.

    """
    def __init__(self, project_folder):
        """
        Arguments
        =========
        project_folder : str
            File path to the project folder

        """
        self._project_folder = project_folder

        reader = SoSModelReader(project_folder)
        builder = SoSModelBuilder()
        reader.builder = builder
        reader.construct()
        self.model = builder.finish()


class SectorConfigReader(object):
    """Parses the ``models/<sector_model>`` folder for a configuration file

    Assign the builder instance to the ``builder`` attribute before running the
    ``construct`` method.

    Arguments
    =========
    model_name : str
        The name of the model
    project_folder : str
        The root path of the project

    """
    def __init__(self, model_name, project_folder):
        self.model_name = model_name
        self.project_folder = project_folder
        self.elements = self.parse_sector_model_config()
        self.builder = None

    def construct(self):
        """Constructs the sector model object from the configuration

        """
        # First, name the model
        self.builder.name_model(self.model_name)
        # Then, load up the wrapper
        self.builder.load_wrapper(self.elements['wrapper'])

        for key, value in self.elements.items():
            if key == 'inputs':
                self.builder.load_inputs(value)
            elif key == 'outputs':
                self.builder.load_outputs(value)
            elif key == 'attributes':
                self.builder.load_attributes(value)

    def parse_sector_model_config(self):
        """Searches the model folder for all the configuration files

        """
        config_path = os.path.join(self.project_folder, 'models',
                                   self.model_name)
        input_path = os.path.join(config_path, 'inputs.yaml')
        output_path = os.path.join(config_path, 'outputs.yaml')
        wrapper_path = os.path.join(config_path, WRAPPER_FILE_NAME)

        assets = self._load_model_assets()
        attribute_paths = {name: os.path.join(self.project_folder, 'models',
                                              self.model_name, 'assets',
                                              "{}.yaml".format(name))
                           for name in assets}

        return {'inputs': input_path,
                'outputs': output_path,
                'attributes': attribute_paths,
                'wrapper': wrapper_path}

    def _load_model_assets(self):
        """Loads the assets from the sector model folders

        Using the list of model folders extracted from the configuration file,
        this function returns a list of all the assets from the sector models

        Returns
        =======
        list
            A list of assets from all the sector models

        """
        assets = []
        path_to_assetfile = os.path.join(self.project_folder,
                                         'models',
                                         self.model_name,
                                         'assets',
                                         'asset*.yaml')
        for assetfile in glob(path_to_assetfile):
            asset_path = os.path.join(path_to_assetfile, assetfile)
            logger.info("Loading assets from {}".format(asset_path))
            assets.extend(ConfigParser(asset_path).data)

        return assets
