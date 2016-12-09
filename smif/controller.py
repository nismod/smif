"""This module coordinates the software components that make up the integration
framework.

"""
import logging
import os
from glob import glob
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
from smif.decision import Planning
from smif.parse_config import ConfigParser
from smif.sectormodel import (SectorModelMode, SectorModelBuilder,
                              SectorConfigReader)

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

LOGGER = logging.getLogger(__name__)


class Controller:
    """Coordinates the data-layer, decision-layer and model-runner

    The Controller expects to be provided with configuration data to run a set
    of sector models over a number of timesteps, in a given mode.

    It also requires a data connection to populate model inputs and store
    results.

    """
    def __init__(self, config_data):
        builder = SosModelBuilder()
        builder.construct(config_data)
        self._model = builder.finish()

    def run_sos_model(self):
        """Runs the system-of-system model
        """
        self._model.run()

    def run_sector_model(self, model_name):
        """Runs a sector model
        """
        self._model.run_sector_model(model_name)



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

    def run(self):
        """Runs the system-of-system model

        1. Determine running order
        2. Run each sector model
        3. Return success or failure
        """
        raise NotImplementedError("Can't run the SOS model yet")

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
        """Runs the sector model

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
        LOGGER.info(msg)

        sector_model = self.model_list[model_name]
        # Run a simulation for a single year
        # TODO fix assumption of no decision vars
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
        """Runs a sequence of simulations to cover each of the model timesteps
        """
        results = []
        for index in range(len(self.timesteps)):
            # Intialise the model
            model.inputs = inputs
            # Update the state from the previous year
            if index > 0:
                state_var = 'existing capacity'
                state_res = results[index - 1]['capacity']
                LOGGER.debug("Updating %s with %s", state_var, state_res)
                model.inputs.parameters.update_value(state_var, state_res)

            # Run the simulation
            decision = decisions[index]
            results.append(model.simulate(decision))
        return results


class SosModelBuilder(object):
    """Constructs a system-of-systems model
    """
    def __init__(self):
        self.sos_model = SosModel()

    def construct(self, config_data):
        """Set up the whole SosModel
        """
        self.add_timesteps(config_data['timesteps'])
        self.load_models(config_data['models'])
        self.add_planning(config_data['planning'])

    def add_timesteps(self, timesteps):
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

    def load_models(self, model_list):
        """Loads the sector models into the system-of-systems model

        Arguments
        =========
        model_list : list
            A list of sector model names

        """
        for model in model_list:
            self.add_model(model)

    def add_model(self, model):
        """Adds a sector model into the system-of-systems model

        """
        LOGGER.info("Loading model: %s", model.name)
        self.sos_model.model_list[model.name] = model

    def add_planning(self, planning):
        """Loads the planning logic into the system of systems model
        """
        self.sos_model.planning = planning

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


    def check_dependencies(self):
        """For each model, compare dependency list of from_models
        against list of available models
        """
        models_available = self.sos_model.sector_models
        for model_name, model in self.sos_model.model_list.items():
            for dep in model.inputs.dependencies:
                if dep.from_model not in models_available:
                    # report missing dependency type
                    msg = "Missing dependency: {} depends on {} from {}, which is not supplied."
                    raise AssertionError(msg.format(model_name, dep.name, dep.from_model))

    def finish(self):
        """Returns a configured system-of-systems model ready for operation

        - includes validation steps, e.g. to check dependencies
        """
        self.validate()
        self.check_dependencies()
        return self.sos_model
