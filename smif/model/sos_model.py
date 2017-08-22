# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

"""
import logging
from collections import defaultdict

import networkx
import numpy as np
from enum import Enum
from smif import StateData
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
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.max_iterations = 25
        self.convergence_relative_tolerance = 1e-05
        self.convergence_absolute_tolerance = 1e-08

        # models - includes types of SectorModel and ScenarioModel
        self.models = {}
        self.dependency_graph = networkx.DiGraph()

        # systems, interventions and (system) state
        self.timesteps = []
        self.interventions = InterventionRegister()
        self.initial_conditions = []
        self.planning = Planning([])
        self._state = defaultdict(dict)

        # scenario data and results
        self._results = defaultdict(dict)

    @property
    def free_inputs(self):
        """Returns the free inputs not linked to a dependency at this layer and all contained layers

        Free inputs are passed up to higher layers for deferred linkages to
        dependencies.

        Returns
        -------
        smif.metadata.MetadataSet
        """
        # free inputs of all contained models
        free_inputs = []
        for model in self.models.values():
            free_inputs.extend(model.free_inputs)

        # free inputs of current layer
        my_free_inputs = super().free_inputs
        free_inputs.extend(my_free_inputs)

        # compose a new MetadataSet containing the free inputs
        metadataset = MetadataSet([])
        for meta in free_inputs:
            metadataset.add_metadata_object(meta)

        return metadataset

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
        self.check_dependencies()
        run_order = self._get_model_sets_in_run_order()
        self.logger.info("Determined run order as %s", run_order)
        results = {}
        for model in run_order:
            sim_results = model.simulate(timestep, data)
            for model_name, model_results in sim_results.items():
                results[model_name] = model_results
        return results

    def check_dependencies(self):
        """For each contained model, compare dependency list against
        list of available models
        """
        if self.free_inputs.names:
            msg = "A SosModel must have all inputs linked to dependencies." \
                  "Define dependencies for %s"
            raise NotImplementedError(msg, ", ".join(self.free_inputs.names))

        for model in self.models.values():

            if isinstance(model, SosModel):
                msg = "Nesting of SosModels not yet supported"
                raise NotImplementedError(msg)
            else:
                self.dependency_graph.add_node(model,
                                               name=model.name)

                for sink, dependency in model.deps.items():
                    provider = dependency.source_model
                    msg = "Dependency '%s' provided by '%s'"
                    self.logger.debug(msg, sink, provider.name)

                    self.dependency_graph.add_edge(provider,
                                                   model,
                                                   {'source': dependency.source,
                                                    'sink': sink})

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

    def _get_model_sets_in_run_order(self):
        """Returns a list of :class:`Model` in a runnable order.

        If a set contains more than one model, there is an interdependency and
        and we attempt to run the models to convergence.
        """
        if networkx.is_directed_acyclic_graph(self.dependency_graph):
            # topological sort gives a single list from directed graph, currently
            # ignoring opportunities to run independent models in parallel
            run_order = networkx.topological_sort(self.dependency_graph, reverse=False)

            # list of Models (typically ScenarioModel and SectorModel)
            ordered_sets = list(run_order)

        else:
            # contract the strongly connected components (subgraphs which
            # contain cycles) into single nodes, producing the 'condensation'
            # of the graph, where each node maps to one or more sector models
            condensation = networkx.condensation(self.dependency_graph)

            # topological sort of the condensation gives an ordering of the
            # contracted nodes, whose 'members' attribute refers back to the
            # original dependency graph
            ordered_sets = []
            for node_id in networkx.topological_sort(condensation, reverse=False):
                models = condensation.node[node_id]['members']
                if len(models) == 1:
                    ordered_sets.append(models[0])
                else:
                    ordered_sets.append(ModelSet(models))

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
        return element_before(timestep, self.timesteps)

    def timestep_after(self, timestep):
        """Returns the timestep after a given timestep, or None
        """
        return element_after(timestep, self.timesteps)

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


class ModelSet(Model):
    """Wraps a set of interdependent models

    Given a directed graph of dependencies between models, any cyclic
    dependencies are contained within the strongly-connected components of the
    graph.

    A ModelSet corresponds to the set of models within a single strongly-
    connected component. This class provides the machinery necessary
    to find a solution to each of the interdependent models.

    The current implementation first estimates the outputs for each model in the
    set, guaranteeing that each model will then be able to run, then begins
    iterating, running every model in the set at each iteration, monitoring the
    model outputs over the iterations, and stopping at timeout, divergence or
    convergence.

    Arguments
    ---------
    models : list
        A list of smif.model.composite.Model

    """
    def __init__(self, models, max_iterations=25, relative_tolerance=1e-05,
                 absolute_tolerance=1e-08):
        self.logger = logging.getLogger(__name__)
        self._models = models
        self._model_names = {model.name for model in models}
        self.timestep = None
        self.iterated_results = None
        self.max_iterations = max_iterations
        # tolerance for convergence assessment - see numpy.allclose docs
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def simulate(self, timestep, data=None):
        """Runs a set of one or more models

        Arguments
        ---------
        timestep : int
        data : dict, default=None
        """
        # Start by running all models in set with best guess
        # - zeroes
        # - last year's inputs
        self.iterated_results = [{}]
        self.timestep = timestep
        for model in self._models:
            results = self.guess_results(model, timestep, data)
            self.iterated_results[-1][model.name] = results

        # - keep track of intermediate results (iterations within the timestep)
        # - stop iterating according to near-equality condition
        for i in range(self.max_iterations):
            if self.converged():
                break
            else:
                self._run_iteration(i)
        else:
            raise TimeoutError("Model evaluation exceeded max iterations")

        return self.get_last_iteration_results()

    def _run_iteration(self, i):
        """Run all models within the set
        """
        self.iterated_results.append({})
        for model in self._models:
            data = {}
            for model_input, dep in model.deps.items():
                # if internal dependency
                data[model_input] = \
                    self.iterated_results[-2][dep.source_model][dep.source.name]
                # else, pull from data provided to this ModelSet
                # TODO
            results = model.simulate(self.timestep, data)

            self.logger.debug("Iteration %s, model %s, results: %s",
                              i, model.name, results)
            self.iterated_results[-1][model.name] = results

    def get_last_iteration_results(self):
        """Return results from the last iteration

        Returns
        -------
        results : dict
            Dictionary of Model results, keyed by model name
        """
        return self.iterated_results[-1]

    def guess_results(self, model, timestep, data):
        """Dependency-free guess at a model's result set.

        Initially, guess zeroes, or the previous timestep's results.
        """
        if data is None:
            data = {}

        timesteps = sorted(list(data.keys()))
        timestep_before = element_before(timestep, timesteps)
        if timestep_before is not None:
            # last iteration of previous timestep results
            results = data[timestep_before][model.name]
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
        if len(self.iterated_results) < 2:
            # must have at least two result sets per model to assess convergence
            return False

        # iterated_results is a list of dicts with
        #   str key (model name) =>
        #       list of data output from models
        #
        # each data output is a dict with
        #   str key (parameter name) =>
        #       np.ndarray value (regions x intervals)
        if all(
                self._model_converged(
                    model,
                    self.iterated_results[-1][model.name],
                    self.iterated_results[-2][model.name])
                for model in self._models):
            # if all most recent are almost equal to penultimate, must have converged
            return True

        # TODO check for divergance and raise error

        return False

    def _get_model(self, model_name):
        model = [model for model in self._models if model.name == model_name]
        return model[0]

    def _model_converged(self, model, latest_results, previous_results):
        """Check a single model's output for convergence

        Compare data output for each param over recent iterations.

        Parameters
        ----------
        model: Model
        latest_results: dict
            dict of data output from model with str key (parameter name) =>
            np.ndarray value (with dimensions regions x intervals)
        previous_results: dict
            dict of data output from the model in the previous iteration
        """
        return all(
            np.allclose(
                latest_results[param.name],
                previous_results[param.name],
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance
            )
            for param in model.model_outputs.metadata
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
        self.add_dependencies(config_data['dependencies'])

    def add_dependencies(self, dependency_list):
        """Add dependencies between models

        Arguments
        ---------
        dependency_list : list
            A list of dicts of dependency configuration data

        Examples
        --------
        >>> dependencies = [{'source_model': 'raininess',
                             'source_model_output': 'raininess',
                             'sink_model': 'water_supply',
                             'sink_model_input': 'raininess'}]
        >>> builder.add_dependencies(dependencies)
        """
        for dep in dependency_list:
            sink_model_object = self.sos_model.models[dep['sink_model']]
            source_model_object = self.sos_model.models[dep['source_model']]
            source_model_output = dep['source_model_output']
            sink_model_input = dep['sink_model_input']

            self.logger.debug("Adding dependency linking %s.%s to %s.%s",
                              source_model_object.name, source_model_output,
                              sink_model_object.name, sink_model_input)

            sink_model_object.add_dependency(source_model_object,
                                             source_model_output,
                                             sink_model_input)

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
            self.add_interventions(model_data['name'],
                                   model_data['interventions'])
            self.sos_model.add_model(model)
            self.add_model_data(model, model_data)

    def load_scenario_models(self, scenario_list, scenario_data, timesteps):
        """Loads the scenario models into the system-of-systems model

        Note that we currently use the same name for the scenario name,
        and the name of the output of the ScenarioModel.

        Arguments
        ---------
        scenario_list : list
            A list of dicts with keys::

                'name': 'mass',
                'spatial_resolution': 'country',
                'temporal_resolution': 'seasonal',
                'units': 'kg'

        scenario_data : dict
            A dict-of-list-of-dicts with keys ``param_name``: ``year``,
            ``region``, ``interval``, ``value``
        timesteps : list

        Example
        -------
        >>> builder = SosModelBuilder('test_sos_model')
        >>> model_list = [{'name': 'mass',
                           'spatial_resolution': 'country',
                           'temporal_resolution': 'seasonal',
                           'units': 'kg'}]
        >>> data = {'mass': [{'year': 2015,
                              'region': 'GB',
                              'interval': 'wet_season',
                              'value': 3}]}
        >>> timesteps = [2015, 2016]
        >>> builder.load_scenario_models(model_list, data, timesteps)

        """
        self.logger.info("Loading scenarios")
        for scenario_meta in scenario_list:
            name = scenario_meta['name']

            if name not in scenario_data:
                msg = "Parameter '{}' in scenario definitions not registered in scenario data"
                raise ValueError(msg.format(name))

            scenario = ScenarioModel(name)

            spatial = scenario_meta['spatial_resolution']
            temporal = scenario_meta['temporal_resolution']

            spatial_res = self.region_register.get_entry(spatial)
            temporal_res = self.interval_register.get_entry(temporal)

            scenario.add_output(name,
                                spatial_res,
                                temporal_res,
                                scenario_meta['units'])

            data = self._data_list_to_array(name,
                                            scenario_data[name],
                                            timesteps,
                                            spatial_res,
                                            temporal_res)
            scenario.add_data(data, timesteps)
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
                            spatial_resolution, temporal_resolution):
        """Convert list of observations to :class:`numpy.ndarray`

        Arguments
        ---------
        param : str
        observations : list
        timestep_names : list
        spatial_resolution : smif.convert.area.RegionSet
        temporal_resolution : smif.convert.interval.IntervalSet

        """
        interval_names = temporal_resolution.get_entry_names()
        region_names = spatial_resolution.get_entry_names()

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
                raise ValueError("Scenario data item missing year: '{}'".format(obs))
            year = obs['year']
            if year not in timestep_names:
                raise ValueError(
                    "Year '{}' not defined in model timesteps".format(year))

            if 'region' not in obs:
                raise ValueError("Scenario data item missing region: '{}'".format(obs))
            region = obs['region']
            if region not in region_names:
                raise ValueError(
                    "Region '{}' not defined in set '{}' for parameter '{}'".format(
                        region,
                        spatial_resolution.name,
                        param))

            if 'interval' not in obs:
                raise ValueError("Scenario data item missing interval: {}".format(obs))
            interval = obs['interval']
            if interval not in interval_names:
                raise ValueError(
                    "Interval '{}' not defined in set '{}' for parameter '{}'".format(
                        interval,
                        temporal_resolution.name,
                        param))

            timestep_idx = timestep_names.index(year)
            interval_idx = interval_names.index(interval)
            region_idx = region_names.index(region)

            data[timestep_idx, region_idx, interval_idx] = obs['value']

        return data

    def _check_planning_interventions_exist(self):
        """Check existence of all the interventions in the pre-specifed planning

        """
        model = self.sos_model
        names = model.intervention_names
        for planning_name in model.planning.names:
            msg = "Intervention '{}' in planning file not found in interventions"
            assert planning_name in names, msg.format(planning_name)

    def _validate(self):
        """Validates the sos model
        """
        self._check_planning_interventions_exist()

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


def element_before(element, list_):
    """Return the element before a given element in a list, or None if the
    given element is first or not in the list.
    """
    if element not in list_ or element == list_[0]:
        return None
    else:
        index = list_.index(element)
        return list_[index - 1]


def element_after(element, list_):
    """Return the element after a given element in a list, or None if the
    given element is last or not in the list.
    """
    if element not in list_ or element == list_[-1]:
        return None
    else:
        index = list_.index(element)
        return list_[index + 1]
