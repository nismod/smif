# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

"""
import logging
import operator
from collections import defaultdict

import networkx
from enum import Enum
from smif import SpaceTimeValue
from smif.convert import SpaceTimeConvertor
from smif.convert.area import RegionRegister, RegionSet
from smif.convert.interval import TimeIntervalRegister
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
        self.regions = RegionRegister()
        self.intervals = TimeIntervalRegister()

        self.planning = Planning([])
        self._scenario_data = {}

        self.logger = logging.getLogger(__name__)

        self.dependency_graph = None

        self._results = defaultdict(dict)

        self._resolution_mapping = {'scenario': {}}

    @property
    def resolution_mapping(self):
        """Returns the temporal and spatial mapping to an input, output or scenario parameter

        Example
        -------
        The data structure follows ``source->parameter->{temporal, spatial}``::

                {'scenario': {
                 'raininess': {'temporal_resolution': 'annual',
                               'spatial_resolution': 'LSOA'}}}
        """
        return self._resolution_mapping

    @resolution_mapping.setter
    def resolution_mapping(self, value):
        self._resolution_mapping = value

    @property
    def scenario_data(self):
        """Get nested dict of scenario data

        Returns
        -------
        dict
            Nested dictionary in the format ``data[year][param] =
            SpaceTimeValue(region, interval, value, unit)``
        """
        return self._scenario_data

    @property
    def results(self):
        """Get nested dict of model results

        Returns
        -------
        dict
            Nested dictionary in the format
            results[int:year][str:model][str:parameter] => list of
            SpaceTimeValues
        """
        # convert from defaultdict to plain dict
        return dict(self._results)

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
        decisions = self._get_decisions(model, timestep)
        state = {}
        data = self._get_data(model, model.name, timestep)
        results = model.simulate(decisions, state, data)
        self.logger.debug("Results from %s model:\n %s", model.name, results)
        self._results[timestep][model.name] = results

    def _get_decisions(self, model, timestep):
        """Gets the interventions that correspond to the decisions

        Parameters
        ----------
        model: :class:`smif.sector_model.SectorModel`
            The instance of the sector model wrapper to run
        timestep: int
            The current model year
        """
        self.logger.debug("Finding decisions for %i", timestep)
        current_decisions = []
        for decision in self.planning.planned_interventions:
            if decision['build_date'] <= timestep:
                name = decision['name']
                if name in model.intervention_names:
                    msg = "Adding decision '%s' to instruction list"
                    self.logger.debug(msg, name)
                    intervention = self.interventions.get_intervention(name)
                    current_decisions.append(intervention)
        # for decision in self.planning.get_rule_based_interventions(timestep):
        #   current_decisions.append(intervention)
        # for decision in self.planning.get_optimised_interventions(timestep):
        #   current_decisions.append(intervention)

        return current_decisions

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
        new_data = {}
        for dependency in model.inputs.parameters:
            name = dependency.name
            provider = self.outputs[name]

            for source in provider:

                self.logger.debug("Getting dependency data for '%s' from '%s'",
                                  name, source)

                if source == 'scenario':
                    from_data = self.scenario_data[timestep][name]
                    scenario_map = self.resolution_mapping['scenario']
                    from_spatial_resolution = scenario_map[name]['spatial_resolution']
                    from_temporal_resolution = scenario_map[name]['temporal_resolution']
                    self.logger.debug("Found data: %s", from_data)

                elif source in self.model_list:
                    source_model = self.model_list[source]
                    from_data = self.results[timestep][source][name]
                    from_spatial_resolution = source_model.outputs.get_spatial_res(name)
                    from_temporal_resolution = source_model.outputs.get_temporal_res(name)
                    self.logger.debug("Found data: %s", from_data)

                else:
                    msg = "The data source for dependency %s was not found"
                    raise ValueError(msg, name)

                to_spatial_resolution = dependency.spatial_resolution
                to_temporal_resolution = dependency.temporal_resolution
                msg = "Converting from spatial resolution '%s' and  temporal resolution '%s'"
                self.logger.debug(msg, from_spatial_resolution, from_temporal_resolution)
                msg = "Converting to spatial resolution '%s' and  temporal resolution '%s'"
                self.logger.debug(msg, to_spatial_resolution, to_temporal_resolution)

                if name not in new_data:
                    new_data[name] = self._convert_data(from_data,
                                                        to_spatial_resolution,
                                                        to_temporal_resolution,
                                                        from_spatial_resolution,
                                                        from_temporal_resolution)
                else:
                    to_add = self._convert_data(from_data,
                                                to_spatial_resolution,
                                                to_temporal_resolution,
                                                from_spatial_resolution,
                                                from_temporal_resolution)
                    new_data[name] = SosModel.add_data_series(new_data[name], to_add)

        new_data['timestep'] = timestep
        return new_data

    def _convert_data(self, data, to_spatial_resolution,
                      to_temporal_resolution, from_spatial_resolution,
                      from_temporal_resolution):
        """Given a model, check required parameters, pick data from scenario
        for the given timestep

        Parameters
        ----------
        timestep: int
            The year for which to get scenario data
        dependency: :class:`smif.SpaceTimeValue`

        Returns
        -------
        list
            A list of :class:`SpaceTimeValue`

        """
        convertor = SpaceTimeConvertor(data,
                                       from_spatial_resolution,
                                       to_spatial_resolution,
                                       from_temporal_resolution,
                                       to_temporal_resolution,
                                       self.regions,
                                       self.intervals)
        return convertor.convert()

    @staticmethod
    def add_data_series(list_a, list_b):
        """Given two lists of SpaceTimeValues of identical spatial and temporal
        resolution, return a single list with matching values added together.

        Notes
        -----
        Assumes a data series is not sparse, i.e. has a value for every
        region/interval combination
        """
        list_a.sort(key=operator.attrgetter('region', 'interval'))
        list_b.sort(key=operator.attrgetter('region', 'interval'))

        return [a + b for a, b in zip(list_a, list_b)]

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

        run_order = networkx.topological_sort(self.dependency_graph,
                                              reverse=True)
        self.logger.debug("Running models in order: %s", run_order)
        return run_order

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

    @property
    def inputs(self):
        """A dictionary of model names associated with an inputs

        Returns
        -------
        dict
            Keys are parameter names, value is a list of sector model names
        """
        parameter_model_map = defaultdict(list)
        for model_name, model_data in self.model_list.items():
            for dep in model_data.inputs.parameters:
                parameter_model_map[dep.name].append(model_name)
        return parameter_model_map

    @property
    def outputs(self):
        """Model names associated with model outputs & scenarios

        Returns
        -------
        dict
            Keys are parameter names, value is a list of sector model names
        """
        parameter_model_map = defaultdict(list)
        for model_name, model_data in self.model_list.items():
            for output in model_data.outputs.parameters:
                parameter_model_map[output.name].append(model_name)

        for name in self.resolution_mapping['scenario'].keys():
            parameter_model_map[name].append('scenario')
        return parameter_model_map


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

        self.load_region_sets(config_data['region_sets'])
        self.load_interval_sets(config_data['interval_sets'])

        self.load_models(model_list)
        self.add_planning(config_data['planning'])
        self.add_resolution_mapping(config_data['resolution_mapping'])
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

    def add_resolution_mapping(self, resolution_mapping):
        """

        Parameters
        ----------
        resolution_mapping: dict
            A dictionary containing information on the spatial and temporal
            resoultion of scenario data

        Example
        -------
        The data structure follows ``source->parameter->{temporal, spatial}``::

                {'scenario': {
                 'raininess': {'temporal_resolution': 'annual',
                               'spatial_resolution': 'LSOA'}}}

        """
        self.sos_model.resolution_mapping = resolution_mapping

    def load_region_sets(self, region_sets):
        """Loads the region sets into the system-of-system model

        Parameters
        ----------
        region_sets: list
            A dict, where key is the name of the region set, and the value
            the data
        """
        assert isinstance(region_sets, dict)

        region_set_definitions = region_sets.items()
        if len(region_set_definitions) == 0:
            msg = "No region sets have been defined"
            self.logger.warning(msg)
        for name, data in region_set_definitions:
            msg = "Region set data is not a list"
            assert isinstance(data, list), msg
            self.sos_model.regions.register(RegionSet(name, data))

    def load_interval_sets(self, interval_sets):
        """Loads the time-interval sets into the system-of-system model

        Parameters
        ----------
        interval_sets: list
            A dict, where key is the name of the interval set, and the value
            the data
        """
        interval_set_definitions = interval_sets.items()
        if len(interval_set_definitions) == 0:
            msg = "No interval sets have been defined"
            self.logger.warning(msg)

        for name, data in interval_set_definitions:
            self.sos_model.intervals.register(data, name)

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
            msg = "Adding %s from %s to SosModel InterventionRegister"
            identifier = intervention_object.name
            self.logger.debug(msg, identifier, model.name)
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

        Expect a dictionary, where each key maps a parameter
        name to a list of data, each observation with:

        - timestep
        - value
        - units
        - region (must use a region id from scenario regions)
        - interval (must use an id from scenario time intervals)

        Add a dictionary of list of :class:`smif.SpaceTimeValue` named
        tuples,
        for ease of iteration::

                data[year][param] = SpaceTimeValue(region, interval, value, units)

        Default region: "national"
        Default interval: "annual"
        """
        self.logger.info("Adding scenario data")
        nested = {}

        for param, observations in data.items():
            if param not in self.sos_model.resolution_mapping['scenario']:
                raise ValueError("Parameter {} not registered in resolution mapping {}".format(
                    param,
                    self.sos_model.resolution_mapping))
            resolution_sets = self.sos_model.resolution_mapping['scenario'][param]

            interval_set_name = resolution_sets['temporal_resolution']
            interval_set = self.sos_model.intervals.get_intervals_in_set(interval_set_name)
            interval_names = [interval.name for key, interval in interval_set.items()]

            region_set_name = resolution_sets['spatial_resolution']
            region_set = self.sos_model.regions.get_regions_in_set(region_set_name)
            region_names = [region.name for region in region_set]

            for obs in observations:
                if 'year' not in obs:
                    raise ValueError("Scenario data item missing year: {}".format(obs))
                year = obs['year']
                if year not in nested:
                    nested[year] = {}

                region = obs['region']
                if region not in region_names:
                    raise ValueError(
                        "Region {} not defined in set {} for parameter {}".format(
                            region,
                            region_set_name,
                            param))

                interval = obs['interval']
                if interval not in interval_names:
                    raise ValueError(
                        "Interval {} not defined in set {} for parameter {}".format(
                            interval,
                            interval_set_name,
                            param))

                entry = SpaceTimeValue(
                    region,
                    interval,
                    obs['value'],
                    obs['units']
                )
                if param not in nested[year]:
                    nested[year][param] = [entry]
                else:
                    nested[year][param].append(entry)

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
        self._check_dependencies()
        self._check_region_interval_sets()

    def _check_region_interval_sets(self):
        """For each model, check for the interval and region sets referenced

        Each model references interval and region sets in the configuration
        of inputs and outputs.
        """
        available_intervals = self.sos_model.intervals.interval_set_names
        msg = "Available time interval sets in SosModel: %s"
        self.logger.debug(msg, available_intervals)
        available_regions = self.sos_model.regions.region_set_names
        msg = "Available region sets in SosModel: %s"
        self.logger.debug(msg, available_regions)

        for model_name, model in self.sos_model.model_list.items():
            exp_regions = []
            exp_intervals = []
            exp_regions.extend(model.inputs.spatial_resolutions)
            exp_regions.extend(model.outputs.spatial_resolutions)
            exp_intervals.extend(model.inputs.temporal_resolutions)
            exp_intervals.extend(model.outputs.temporal_resolutions)

            for region in exp_regions:
                if region not in available_regions:
                    msg = "Region set '%s' not specified but is required " + \
                          "for model '$s'"
                    raise ValueError(msg, region, model_name)
            for interval in exp_intervals:
                if interval not in available_intervals:
                    msg = "Interval set '%s' not specified but is required " + \
                          "for model '$s'"
                    raise ValueError(msg, interval, model_name)

    def _check_dependencies(self):
        """For each model, compare dependency list of from_models
        against list of available models
        """
        dependency_graph = networkx.DiGraph()
        models_available = self.sos_model.sector_models
        dependency_graph.add_nodes_from(models_available)

        for model_name, model in self.sos_model.model_list.items():
            for dep in model.inputs.parameters:
                providers = self.sos_model.outputs[dep.name]
                msg = "Dependency '%s' provided by '%s'"
                self.logger.debug(msg, dep.name, providers)

                if len(providers) == 0:
                    # report missing dependency type
                    msg = "Missing dependency: {} depends on {}, " + \
                        "which is not supplied."
                    raise AssertionError(msg.format(model_name, dep.name))

                for source in providers:
                    if source == 'scenario':
                        continue

                    dependency_graph.add_edge(model_name, source)

        if not networkx.is_directed_acyclic_graph(dependency_graph):
            raise NotImplementedError("Graph of dependencies contains a cycle.")

        self.sos_model.dependency_graph = dependency_graph

    def finish(self):
        """Returns a configured system-of-systems model ready for operation

        Includes validation steps, e.g. to check dependencies
        """
        self._validate()
        return self.sos_model


class RunMode(Enum):
    """Enumerates the operating modes of a SoS model
    """
    static_simulation = 0
    sequential_simulation = 1
    static_optimisation = 2
    dynamic_optimisation = 3
