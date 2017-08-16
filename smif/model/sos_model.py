# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

"""
import logging
from collections import defaultdict
from enum import Enum

import networkx
import numpy as np
from smif import StateData
from smif.convert import SpaceTimeConvertor
from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.decision import Planning
from smif.intervention import Intervention, InterventionRegister
from smif.metadata import MetadataSet
from smif.model.composite import Model
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel, SectorModelBuilder

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SosModel(Model):
    """Consists of the collection of models joined via dependencies

    This class is populated at runtime by the :class:`SosModelBuilder` and
    called from :func:`smif.cli.run_model`.  SosModel inherits from
    :class:`smif.composite.Model`.

    """
    def __init__(self, name):
        # housekeeping
        super().__init__(name, MetadataSet([]), MetadataSet([]))
        self.logger = logging.getLogger(__name__)
        self.max_iterations = 25
        self.convergence_relative_tolerance = 1e-05
        self.convergence_absolute_tolerance = 1e-08

        # models - includes types of SectorModel and ScenarioModel
        self.models = {}
        self.dependency_graph = None

        # systems, interventions and (system) state
        self.timesteps = []
        self.interventions = InterventionRegister()
        self.initial_conditions = []
        self.planning = Planning([])
        self._state = defaultdict(dict)

        # scenario data and results
        self._results = defaultdict(dict)

    def add_model(self, model):
        """Adds a sector model to the system-of-systems model

        Parameters
        ----------
        model : :class:`smif.sector_model.SectorModel`
            A sector model wrapper

        """
        assert isinstance(model, Model)
        self.logger.info("Loading model: %s", model.name)
        self.models[model.name] = model

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

    def simulate(self, timestep, data=None):
        """Run the SosModel

        """
        self._check_dependencies()
        run_order = self._get_model_sets_in_run_order()
        names = [model_set._model_names for model_set in run_order]
        self.logger.info("Determined run order as %s", names)
        for model_set in run_order:
            model_set.run(timestep)
        return self.results

    def _check_dependencies(self):
        """For each model, compare dependency list of from_models
        against list of available models
        """
        dependency_graph = networkx.DiGraph()
        dependency_graph.add_nodes_from(self.models.values())

        for model_name, model in self.models.items():
            if model.model_inputs:
                for model_input in model.model_inputs:
                    self.logger.debug(model_input)
                    if model_input.name in model.deps:
                        dependency = model.deps[model_input.name]
                        provider = dependency.source_model
                        msg = "Dependency '%s' provided by '%s'"
                        self.logger.debug(msg, model_input.name, provider.name)

                        dependency_graph.add_edge(provider,
                                                  model,
                                                  {'source': dependency.source,
                                                   'sink': model_input})
                    else:
                        # report missing dependency type
                        msg = "Missing dependency: '{}' depends on '{}', " + \
                            "which is not supplied."
                        raise AssertionError(msg.format(model_name,
                                                        model_input.name))

        self.dependency_graph = dependency_graph

    def get_decisions(self, model, timestep):
        """Gets the interventions that correspond to the decisions

        Parameters
        ----------
        model: :class:`smif.sector_model.SectorModel`
            The instance of the sector model wrapper to run
        timestep: int
            The current model year

        TODO: Move into DecisionManager class
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

    def get_state(self, model, timestep):
        """Gets the state to pass to SectorModel.simulate
        """
        if model.name not in self._state[timestep]:
            self.logger.warning("Found no state for %s in timestep %s", model.name, timestep)
            return []
        return self._state[timestep][model.name]

    def set_state(self, model, from_timestep, state):
        """Sets state output from model ready for next timestep
        """
        for_timestep = self.timestep_after(from_timestep)
        self._state[for_timestep][model.name] = state

    def set_data(self, model, timestep, results):
        """Sets results output from model as data available to other/future models

        Stores only latest estimated results (i.e. not holding on to iterations
        here while trying to solve interdependencies)
        """
        self._results[timestep][model.name] = results

    def _convert_data(self, data, to_spatial_resolution,
                      to_temporal_resolution, from_spatial_resolution,
                      from_temporal_resolution):
        """Convert data from one spatial and temporal resolution to another

        Parameters
        ----------
        data : numpy.ndarray
            The data series for conversion
        to_spatial_resolution : str
            ID of the region set to convert to
        to_temporal_resolution : str
            ID of the interval set to convert to
        from_spatial_resolution : str
            ID of the region set to convert from
        from_temporal_resolution : str
            ID of the interval set to convert from

        Returns
        -------
        converted_data : numpy.ndarray
            The converted data series

        """
        convertor = SpaceTimeConvertor(self.regions, self.intervals)
        return convertor.convert(data,
                                 from_spatial_resolution,
                                 to_spatial_resolution,
                                 from_temporal_resolution,
                                 to_temporal_resolution)

    def _get_model_sets_in_run_order(self):
        """Returns a list of :class:`ModelSet` in a runnable order.

        If a set contains more than one model, there is an interdependency and
        and we attempt to run the models to convergence.
        """
        if networkx.is_directed_acyclic_graph(self.dependency_graph):
            # topological sort gives a single list from directed graph, currently
            # ignoring opportunities to run independent models in parallel
            run_order = networkx.topological_sort(self.dependency_graph, reverse=False)

            # turn into a list of sets for consistency with the below
            ordered_sets = [
                ModelSet(
                    {model},
                    self
                )
                for model in run_order
            ]

        else:
            # contract the strongly connected components (subgraphs which
            # contain cycles) into single nodes, producing the 'condensation'
            # of the graph, where each node maps to one or more sector models
            condensation = networkx.condensation(self.dependency_graph)

            # topological sort of the condensation gives an ordering of the
            # contracted nodes, whose 'members' attribute refers back to the
            # original dependency graph
            ordered_sets = [
                ModelSet(
                    {
                        model
                        for model in condensation.node[node_id]['members']
                    },
                    self
                )
                for node_id in networkx.topological_sort(condensation, reverse=False)
            ]

        return ordered_sets

    def determine_running_mode(self):
        """Determines from the config in what mode to run the model

        Returns
        =======
        :class:`RunMode`
            The mode in which to run the model
        """

        number_of_timesteps = len(self.timesteps)

        if number_of_timesteps > 1:
            # Run a sequential simulation
            mode = RunMode.sequential_simulation

        elif number_of_timesteps == 0:
            raise ValueError("No timesteps have been specified")

        else:
            # Run a single simulation
            mode = RunMode.static_simulation

        return mode

    def timestep_before(self, timestep):
        """Returns the timestep previous to a given timestep, or None
        """
        if timestep not in self.timesteps or timestep == self.timesteps[0]:
            return None
        else:
            index = self.timesteps.index(timestep)
            return self.timesteps[index - 1]

    def timestep_after(self, timestep):
        """Returns the timestep after a given timestep, or None
        """
        if timestep not in self.timesteps or timestep == self.timesteps[-1]:
            return None
        else:
            index = self.timesteps.index(timestep)
            return self.timesteps[index + 1]

    @property
    def intervention_names(self):
        """Names (id-like keys) of all known asset type
        """
        return [intervention.name for intervention in self.interventions]

    @property
    def sector_models(self):
        """The list of sector model names

        Returns
        =======
        list
            A list of sector model names
        """
        return [x for x, y in self.models.items() if isinstance(y, SectorModel)]

    @property
    def scenario_models(self):
        """The list of scenario model names

        Returns
        -------
        list
            A list of scenario model names
        """
        return [x for x, y in self.models.items()
                if isinstance(y, ScenarioModel)
                ]


class ModelSet(object):
    """Wraps a set of interdependent models

    Given a directed graph of dependencies between models, any cyclic
    dependencies are contained within the strongly-connected components of the
    graph.

    A ModelSet corresponds to the set of models within a single strongly-
    connected component. If this is a set of one model, it can simply be run
    deterministically. Otherwise, this class provides the machinery necessary
    to find a solution to each of the interdependent models.

    The current implementation first estimates the outputs for each model in the
    set, guaranteeing that each model will then be able to run, then begins
    iterating, running every model in the set at each iteration, monitoring the
    model outputs over the iterations, and stopping at timeout, divergence or
    convergence.

    Arguments
    ---------
    models : dict
        A list of smif.model.composite.Model
    sos_model : smif.model.sos_model.SosModel
        A SosModel instance containing the models
    Notes
    -----
    This calls back into :class:`SosModel` quite extensively for state, data,
    decisions, regions and intervals.

    """
    def __init__(self, models, sos_model):
        self.logger = logging.getLogger(__name__)
        self._models = models
        self._model_names = {model.name for model in models}
        self._sos_model = sos_model
        self.iterated_results = {}
        self.max_iterations = sos_model.max_iterations
        # tolerance for convergence assessment - see numpy.allclose docs
        self.relative_tolerance = sos_model.convergence_relative_tolerance
        self.absolute_tolerance = sos_model.convergence_absolute_tolerance

    def run(self, timestep):
        """Runs a set of one or more models
        """
        if len(self._models) == 1:
            # Short-circuit if the set contains a single model - this
            # can be run deterministically
            model = list(self._models)[0]
            logging.debug("Running %s for %d", model.name, timestep)

            data = {}
            if model.model_inputs:
                for model_input in model.model_inputs:
                    self.logger.debug("Seeking dep for %s", model_input.name)
                    if model_input.name not in model.deps:
                        msg = "Dependency not found for '{}'"
                        raise ValueError(msg.format(model_input.name))
                    else:
                        dependency = model.deps[model_input.name]
                        self.logger.debug("Found dependency for '%s'",
                                          model_input.name)
                        data[model_input.name] = dependency.get_data(timestep)

            results = model.simulate(timestep, data)
            # self._sos_model.set_state(model, timestep, state)
            self._sos_model.set_data(model, timestep, results)
        else:
            # Start by running all models in set with best guess
            # - zeroes
            # - last year's inputs
            self.iterated_results = {}
            for model in self._models:
                results = self.guess_results(model, timestep)
                self._sos_model.set_data(model, timestep, results)
                self.iterated_results[model.name] = [results]

            # - keep track of intermediate results (iterations within the timestep)
            # - stop iterating according to near-equality condition
            for i in range(self.max_iterations):
                if self.converged():
                    break
                else:
                    self.logger.debug("Iteration %s, model set %s", i, self._model_names)
                    for model in self._models:
                        results = model.simulate(timestep, data)
                        # self._sos_model.set_state(model, timestep, state)
                        self._sos_model.set_data(model, timestep, results)
                        self.iterated_results[model.name].append(results)
            else:
                raise TimeoutError("Model evaluation exceeded max iterations")

    def guess_results(self, model, timestep):
        """Dependency-free guess at a model's result set.

        Initially, guess zeroes, or the previous timestep's results.
        """
        timestep_before = self._sos_model.timestep_before(timestep)
        if timestep_before is not None:
            # last iteration of previous timestep results
            results = self._sos_model.results[timestep_before][model.name]
        else:
            # generate zero-values for each parameter/region/interval combination
            results = {}
            for output in model.model_outputs.metadata:
                regions = output.get_region_names()
                intervals = output.get_interval_names()
                results[output.name] = np.zeros((len(regions), len(intervals)))
        return results

    def converged(self):
        """Check whether the results of a set of models have converged.

        Returns
        -------
        converged: bool
            True if the results have converged to within a tolerance

        Raises
        ------
        DiverganceError
            If the results appear to be diverging
        """
        if any([len(results) < 2 for results in self.iterated_results.values()]):
            # must have at least two result sets per model to assess convergence
            return False

        # iterated_results is a dict with
        #   str key (model name) =>
        #       list of data output from models
        #
        # each data output is a dict with
        #   str key (parameter name) =>
        #       np.ndarray value (regions x intervals)
        if all(self._model_converged(self._get_model(model_name), results)
               for model_name, results in self.iterated_results.items()):
            # if all most recent are almost equal to penultimate, must have converged
            return True

        # TODO check for divergance and raise error

        return False

    def _get_model(self, model_name):
        model = [model for model in self._models if model.name == model_name]
        return model[0]

    def _model_converged(self, model, results):
        """Check a single model's output for convergence

        Compare data output for each param over recent iterations.

        Parameters
        ----------
        results: list
            list of data output from models, from first iteration. Each list
            entry is a dict with str key (parameter name) => np.ndarray value
            (with dimensions regions x intervals)
        """
        latest_results = results[-1]
        previous_results = results[-2]
        param_names = [param.name for param in model.model_outputs.metadata]

        return all(
            np.allclose(
                latest_results[param],
                previous_results[param],
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance
            )
            for param in param_names
        )


class SosModelBuilder(object):
    """Constructs a system-of-systems model

    Builds a :class:`SosModel`.

    Arguments
    ---------
    name: str, default=''
        The unique name of the SosModel

    Examples
    --------
    Call :py:meth:`SosModelBuilder.construct` to populate
    a :py:class:`SosModel` object and :py:meth:`SosModelBuilder.finish`
    to return the validated and dependency-checked system-of-systems model.

    >>> builder = SosModelBuilder('test_model')
    >>> builder.construct(config_data, timesteps)
    >>> sos_model = builder.finish()

    """
    def __init__(self, name=''):
        self.sos_model = SosModel(name)
        self.region_register = get_region_register()
        self.interval_register = get_interval_register()

        self.logger = logging.getLogger(__name__)

    def construct(self, config_data, timesteps):
        """Set up the whole SosModel

        Parameters
        ----------
        config_data : dict
            A valid system-of-systems model configuration dictionary
        timesteps : list
            A list of timestep integers
        """
        model_list = config_data['sector_model_data']

        self.set_max_iterations(config_data)
        self.set_convergence_abs_tolerance(config_data)
        self.set_convergence_rel_tolerance(config_data)

        self.load_models(model_list)
        self.load_scenario_models(config_data['scenario_metadata'],
                                  config_data['scenario_data'],
                                  timesteps)
        self.add_planning(config_data['planning'])

    def set_max_iterations(self, config_data):
        """Set the maximum iterations for iterating `class`::smif.ModelSet to
        convergence
        """
        if 'max_iterations' in config_data and config_data['max_iterations'] is not None:
            self.sos_model.max_iterations = config_data['max_iterations']

    def set_convergence_abs_tolerance(self, config_data):
        """Set the absolute tolerance for iterating `class`::smif.ModelSet to
        convergence
        """
        if 'convergence_absolute_tolerance' in config_data and \
                config_data['convergence_absolute_tolerance'] is not None:
            self.sos_model.convergence_absolute_tolerance = \
                config_data['convergence_absolute_tolerance']

    def set_convergence_rel_tolerance(self, config_data):
        """Set the relative tolerance for iterating `class`::smif.ModelSet to
        convergence
        """
        if 'convergence_relative_tolerance' in config_data and \
                config_data['convergence_relative_tolerance'] is not None:
            self.sos_model.convergence_relative_tolerance = \
                config_data['convergence_relative_tolerance']

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
            builder = SectorModelBuilder(model_data['name'])
            builder.construct(model_data)
            model = builder.finish()
            self.sos_model.add_interventions(model_data['name'], )
            self.sos_model.add_model(model)
            self.add_model_data(model, model_data)

    def load_scenario_models(self, scenario_list, scenario_data, timesteps):
        """Loads the scenario models into the system-of-systems model

        """
        self.logger.info("Loading scenarios")
        for scenario_meta in scenario_list:
            scenario_output = MetadataSet([])
            scenario_output.add_metadata(scenario_meta)
            name = scenario_meta['name']
            scenario = ScenarioModel(name,
                                     scenario_output)

            data = self._data_list_to_array(
                    name,
                    scenario_data[name],
                    timesteps,
                    scenario_meta
                )

            scenario.add_data(data)
            self.sos_model.add_model(scenario)

    def add_model_data(self, model, model_data):
        """Adds sector model data to the system-of-systems model which is
        convenient to have available at the higher level.
        """
        # self.add_initial_conditions(model.name, model_data['initial_conditions'])
        self.add_interventions(model.name, model_data['interventions'])

    def add_interventions(self, model_name, interventions):
        """Adds interventions for a model
        """
        for intervention in interventions:
            intervention_object = Intervention(sector=model_name,
                                               data=intervention)
            msg = "Adding %s from %s to SosModel InterventionRegister"
            identifier = intervention_object.name
            self.logger.debug(msg, identifier, model_name)
            self.sos_model.interventions.register(intervention_object)

    def add_initial_conditions(self, model_name, initial_conditions):
        """Adds initial conditions (state) for a model
        """
        timestep = self.sos_model.timesteps[0]
        state_data = filter(
            lambda d: len(d.data) > 0,
            [self.intervention_state_from_data(datum) for datum in initial_conditions]
        )
        self.sos_model._state[timestep][model_name] = list(state_data)

    @staticmethod
    def intervention_state_from_data(intervention_data):
        """Unpack an intervention from the initial system to extract StateData
        """
        target = None
        data = {}
        for key, value in intervention_data.items():
            if key == "name":
                target = value

            if isinstance(value, dict) and "is_state" in value and value["is_state"]:
                del value["is_state"]
                data[key] = value

        return StateData(target, data)

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

    def _data_list_to_array(self, param, observations, timestep_names,
                            param_metadata):
        """Convert list of observations to :class:`numpy.ndarray`
        """
        interval_names, region_names = self._get_dimension_names_for_param(
            param_metadata, param)

        if len(timestep_names) == 0:
            self.logger.error("No timesteps found when loading %s", param)

        data = np.zeros((
            len(timestep_names),
            len(region_names),
            len(interval_names)
        ))
        data.fill(np.nan)

        if len(observations) != data.size:
            self.logger.warning(
                "Number of observations is not equal to timesteps x  " +
                "intervals x regions when loading %s", param)

        for obs in observations:
            if 'year' not in obs:
                raise ValueError("Scenario data item missing year: {}".format(obs))
            year = obs['year']
            if year not in timestep_names:
                raise ValueError(
                    "Year {} not defined in model timesteps".format(year))

            if 'region' not in obs:
                raise ValueError("Scenario data item missing region: {}".format(obs))
            region = obs['region']
            if region not in region_names:
                raise ValueError(
                    "Region {} not defined in set {} for parameter {}".format(
                        region,
                        param_metadata.spatial_resolution,
                        param))

            if 'interval' not in obs:
                raise ValueError("Scenario data item missing interval: {}".format(obs))
            interval = obs['interval']
            if interval not in interval_names:
                raise ValueError(
                    "Interval {} not defined in set {} for parameter {}".format(
                        interval,
                        param_metadata.temporal_resolution,
                        param))

            timestep_idx = timestep_names.index(year)
            interval_idx = interval_names.index(interval)
            region_idx = region_names.index(region)

            data[timestep_idx, region_idx, interval_idx] = obs['value']

        return data

    def _get_dimension_names_for_param(self, metadata, param):
        interval_set_name = metadata['temporal_resolution']
        interval_set = self.interval_register.get_entry(interval_set_name)
        interval_names = interval_set.get_entry_names()

        region_set_name = metadata['spatial_resolution']
        region_set = self.region_register.get_entry(region_set_name)
        region_names = region_set.get_entry_names()

        if len(interval_names) == 0:
            self.logger.error("No interval names found when loading %s", param)

        if len(region_names) == 0:
            self.logger.error("No region names found when loading %s", param)

        return interval_names, region_names

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
