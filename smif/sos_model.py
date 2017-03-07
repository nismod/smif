# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

"""
import logging
from enum import Enum

import networkx
from smif.decision import Planning
from smif.intervention import Intervention, InterventionRegister
from smif.sector_model import SectorModelBuilder

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


class SosModel(object):
    """Consists of the collection of timesteps and sector models

    This is NISMOD - i.e. the system of system model which brings all of the
    sector models together. Sector models may be joined through dependencies.

    This class is populated at runtime by the :class:`SosModelBuilder` and
    called from :func:`smif.cli.run_model`.

    Attributes
    ==========
    model_list : dict
        This is a dictionary of :class:`smif.SectorModel`

    """
    def __init__(self):
        self.model_list = {}
        self._timesteps = []
        self.initial_conditions = []
        self.interventions = InterventionRegister()
        self.planning = Planning([])
        self._scenario_data = {}

        self.logger = logging.getLogger(__name__)

    @property
    def scenario_data(self):
        """Get nested dict of scenario data

        Returns
        -------
        dict
            Nested dictionary in the format data[year][param][region][interval]
        """
        return self._scenario_data

    def run(self):
        """Runs the system-of-system model

        0. Determine run mode

        1. Determine running order

        2. Run each sector model

        3. Return success or failure
        """
        mode = self.determine_running_mode()
        self.logger.debug("Running in %s mode", mode.name)

        if mode == RunMode.static_simulation:
            self._run_static_sos_model()
        elif mode == RunMode.sequential_simulation:
            self._run_sequential_sos_model()
        elif mode == RunMode.static_optimisation:
            self._run_static_optimisation()
        elif mode == RunMode.dynamic_optimisation:
            self._run_dynamic_optimisation()

    def _run_static_sos_model(self):
        """Runs the system-of-system model for one timeperiod

        Calls each of the sector models in the order required by the graph of
        dependencies, passing in the year for which they need to run.

        """
        run_order = self._get_model_names_in_run_order()
        timestep = self.timesteps[0]

        for model_name in run_order:
            logging.debug("Running %s", model_name)
            sector_model = self.model_list[model_name]
            self._run_sector_model_timestep(sector_model, timestep)

    def _run_sequential_sos_model(self):
        """Runs the system-of-system model sequentially

        """
        run_order = self._get_model_names_in_run_order()
        self.logger.info("Determined run order as %s", run_order)
        for timestep in self.timesteps:
            for model_name in run_order:
                logging.debug("Running %s for %d", model_name, timestep)
                sector_model = self.model_list[model_name]
                self._run_sector_model_timestep(sector_model, timestep)

    def run_sector_model(self, model_name):
        """Runs the sector model

        Parameters
        ----------
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        msg = "Model '{}' does not exist. Choose from {}"
        assert model_name in self.model_list, \
            msg.format(model_name, self.sector_models)

        msg = "Running the %s sector model"
        self.logger.info(msg, model_name)

        sector_model = self.model_list[model_name]

        # Run a simulation for a single model
        for timestep in self.timesteps:
            self._run_sector_model_timestep(sector_model, timestep)

    def _run_sector_model_timestep(self, model, timestep):
        """Run the sector model for a specific timestep

        Parameters
        ----------
        model: :class:`smif.sector_model.SectorModel`
            The instance of the sector model wrapper to run
        timestep: int
            The year for which to run the model

        """
        decisions = []
        state = {}
        data = self._get_data(model, model.name, timestep)
        model.simulate(decisions, state, data)

    def _get_data(self, model, model_name, timestep):
        """Gets the data in the required format to pass to the simulate method

        Returns
        -------
        dict
            A nested dictionary of the format:
            ``data[parameter][region][time_interval] = {value, units}``

        Notes
        -----
        Note that the timestep is `not` passed to the SectorModel in the
        nested data dictionary.
        The current timestep is available in ``data['timestep']``.

        """
        data = {}
        for dependency in model.inputs.dependencies:
            self.logger.debug("Finding data for dependency: %s", dependency.name)
            if dependency.from_model == 'scenario':
                data = self._get_scenario_data(timestep)
                self.logger.debug("Found data: %s", data)
            else:
                msg = "Getting data from dependencies is not yet implemented"
                raise NotImplementedError(msg)
        return data

    def _get_scenario_data(self, timestep):
        """Given a model, check required parameters, pick data from scenario
        for the given timestep

        Parameters
        ----------
        timestep: int
            The year for which to get scenario data

        """
        return self.scenario_data[timestep]

    def _run_static_optimisation(self):
        """Runs the system-of-systems model in a static optimisation format
        """
        raise NotImplementedError

    def _run_dynamic_optimisation(self):
        """Runs the system-of-system models in a dynamic optimisation format
        """
        raise NotImplementedError

    def _get_model_names_in_run_order(self):
        # topological sort gives a single list from directed graph
        return networkx.topological_sort(self.dependency_graph)

    def determine_running_mode(self):
        """Determines from the config in what mode to run the model

        Returns
        =======
        :class:`RunMode`
            The mode in which to run the model
        """

        number_of_timesteps = len(self._timesteps)

        if number_of_timesteps > 1:
            # Run a sequential simulation
            mode = RunMode.sequential_simulation

        elif number_of_timesteps == 0:
            raise ValueError("No timesteps have been specified")

        else:
            # Run a single simulation
            mode = RunMode.static_simulation

        return mode

    @property
    def timesteps(self):
        """Returns the list of timesteps

        Returns
        =======
        list
            A list of timesteps
        """
        return sorted(self._timesteps)

    @property
    def intervention_names(self):
        """Names (id-like keys) of all known asset type
        """
        return [intervention.name for intervention in self.interventions]

    @timesteps.setter
    def timesteps(self, value):
        self._timesteps = value

    @property
    def sector_models(self):
        """The list of sector model names

        Returns
        =======
        list
            A list of sector model names
        """
        return list(self.model_list.keys())


class SosModelBuilder(object):
    """Constructs a system-of-systems model

    Builds a :class:`SosModel`.

    Examples
    --------
    Call :py:meth:`SosModelBuilder.construct` to populate
    a :class:`SosModel` object and :py:meth:`SosModelBuilder.finish`
    to return the validated and dependency-checked system-of-systems model.

    >>> builder = SosModelBuilder()
    >>> builder.construct(config_data)
    >>> sos_model = builder.finish()

    """
    def __init__(self):
        self.sos_model = SosModel()

        self.logger = logging.getLogger(__name__)

    def construct(self, config_data):
        """Set up the whole SosModel

        Parameters
        ----------
        config_data : dict
            A valid system-of-systems model configuration dictionary
        """
        model_list = config_data['sector_model_data']

        self.add_timesteps(config_data['timesteps'])
        self.load_models(model_list)
        self.add_planning(config_data['planning'])
        self.add_scenario_data(config_data['scenario_data'])
        self.logger.debug(config_data['scenario_data'])

    def add_timesteps(self, timesteps):
        """Set the timesteps of the system-of-systems model

        Parameters
        ----------
        timesteps : list
            A list of timesteps
        """
        self.logger.info("Adding timesteps")
        self.sos_model.timesteps = timesteps

    def load_models(self, model_data_list):
        """Loads the sector models into the system-of-systems model

        Parameters
        ----------
        model_data_list : list
            A list of sector model config/data
        assets : list
            A list of assets to pass to the sector model

        """
        self.logger.info("Loading models")
        for model_data in model_data_list:
            model = self._build_model(model_data)
            self.add_model(model)

    @staticmethod
    def _build_model(model_data):
        builder = SectorModelBuilder(model_data['name'])
        builder.load_model(model_data['path'], model_data['classname'])
        builder.add_inputs(model_data['inputs'])
        builder.add_outputs(model_data['outputs'])
        builder.add_assets(model_data['initial_conditions'])
        builder.add_interventions(model_data['interventions'])
        return builder.finish()

    def add_model(self, model):
        """Adds a sector model to the system-of-systems model

        Parameters
        ----------
        model : :class:`smif.sector_model.SectorModel`
            A sector model wrapper

        """
        self.logger.info("Loading model: %s", model.name)
        self.sos_model.model_list[model.name] = model

        for intervention in model.interventions:
            intervention_object = Intervention(sector=model.name,
                                               data=intervention)
            msg = "Adding {} from {} to SOSModel InterventionRegister"
            identifier = intervention_object.name
            self.logger.debug(msg.format(identifier, model.name))
            self.sos_model.interventions.register(intervention_object)

    def add_planning(self, planning):
        """Loads the planning logic into the system of systems model

        Pre-specified planning interventions are defined at the sector-model
        level, read in through the SectorModel class, but populate the
        intervention register in the controller.

        Parameters
        ----------
        planning : list
            A list of planning instructions

        """
        self.logger.info("Adding planning")
        self.sos_model.planning = Planning(planning)

    def add_scenario_data(self, data):
        """Load the scenario data into the system of systems model

        Expect a dictionary, where each key maps a parameter name to a list of
        data, each observation with:
        - year
        - value
        - units
        - region (optional, must use a region id from scenario regions)
        - interval (optional, must use an id from scenario time intervals)

        Add a dictionary re-rolled for ease of lookup:
            data[year][param][region][interval] => {value, units}

        Default region: "UK"
        Default interval: "year"
        """
        self.logger.info("Adding scenario data")
        nested = {}
        for param, observations in data.items():
            for obs in observations:
                if "year" not in obs:
                    raise ValueError("Scenario data item missing year: %s", obs)
                else:
                    year = obs["year"]
                if year not in nested:
                    nested[year] = {}

                if param not in nested[year]:
                    nested[year][param] = {}

                if "region" not in obs:
                    obs["region"] = "UK"
                region = obs["region"]

                if region not in nested[year][param]:
                    nested[year][param][region] = {}

                if "interval" not in obs:
                    obs["interval"] = "year"
                interval = obs["interval"]

                if interval in nested[year][param][region]:
                    raise AssertionError(
                        "Scenario data item duplicated for year, parameter, region: %s, %s",
                        obs,
                        nested[year][param][region][interval]
                    )
                else:
                    del obs["year"]
                    del obs["region"]
                    del obs["interval"]
                    nested[year][param][region][interval] = obs
        self.logger.debug("Added scenario data: %s", nested)
        self.sos_model._scenario_data = nested

    def _check_planning_interventions_exist(self):
        """Check existence of all the interventions in the pre-specifed planning

        """
        model = self.sos_model
        names = model.intervention_names
        for planning_name in model.planning.names:
            msg = "Intervention '{}' in planning file not found in interventions"
            assert planning_name in names, msg.format(planning_name)

    def _check_planning_timeperiods_exist(self):
        """Check existence of all the timeperiods in the pre-specified planning
        """
        model = self.sos_model
        model_timeperiods = model.timesteps
        for timeperiod in model.planning.timeperiods:
            msg = "Timeperiod '{}' in planning file not found model config"
            assert timeperiod in model_timeperiods, msg.format(timeperiod)

    def _validate(self):
        """Validates the sos model
        """
        self._check_planning_interventions_exist()
        self._check_planning_timeperiods_exist()

    def _check_dependencies(self):
        """For each model, compare dependency list of from_models
        against list of available models
        """
        dependency_graph = networkx.DiGraph()
        models_available = self.sos_model.sector_models
        dependency_graph.add_nodes_from(models_available)

        for model_name, model in self.sos_model.model_list.items():
            for dep in model.inputs.dependencies:
                msg = "Dependency '%s' provided by '%s'"
                self.logger.debug(msg, dep.name, dep.from_model)
                if dep.from_model == "scenario":
                    continue

                if dep.from_model not in models_available:
                    # report missing dependency type
                    msg = "Missing dependency: {} depends on {} from {}, " + \
                          "which is not supplied."
                    raise AssertionError(msg.format(model_name, dep.name, dep.from_model))

                dependency_graph.add_edge(model_name, dep.from_model)

        if not networkx.is_directed_acyclic_graph(dependency_graph):
            raise NotImplementedError("Graph of dependencies contains a cycle.")

        self.sos_model.dependency_graph = dependency_graph

    def finish(self):
        """Returns a configured system-of-systems model ready for operation

        - includes validation steps, e.g. to check dependencies
        """
        self._validate()
        self._check_dependencies()
        return self.sos_model


class RunMode(Enum):
    """Enumerates the operating modes of a SoS model
    """
    static_simulation = 0
    sequential_simulation = 1
    static_optimisation = 2
    dynamic_optimisation = 3
