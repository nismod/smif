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
from smif.convert.area import RegionRegister, RegionSet
from smif.convert.interval import TimeIntervalRegister
from smif.decision import Planning
from smif.intervention import Intervention, InterventionRegister
from smif.metadata import MetadataSet
from smif.sector_model import SectorModelBuilder
from smif.state import State

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SosModel(object):
    """Consists of the collection of timesteps and sector models

    This is NISMOD - i.e. the system of system model which brings all of the
    sector models together. Sector models may be joined through dependencies.

    This class is populated at runtime by the :class:`SosModelBuilder` and
    called from :func:`smif.cli.run_model`.

    Attributes
    ==========
    models : dict
        This is a dictionary of :class:`smif.SectorModel`
    initial_conditions : list
        List of interventions required to set up the initial system, with any
        state attributes provided here too

    """
    def __init__(self):
        # housekeeping
        self.logger = logging.getLogger(__name__)
        self.max_iterations = 25
        self.convergence_relative_tolerance = 1e-05
        self.convergence_absolute_tolerance = 1e-08

        # models
        self.models = {}
        self.dependency_graph = None

        # space and time
        self._timesteps = []
        self.regions = RegionRegister()
        self.intervals = TimeIntervalRegister()
        self._scenario_metadata = MetadataSet({})

        # systems, interventions and (system) state
        self.initial_conditions = []
        self.state = None

        # scenario data and results
        self._scenario_data = {}
        self._results = defaultdict(dict)

    @property
    def scenario_metadata(self):
        """Returns the temporal and spatial mapping to an input, output or scenario parameter
        """
        return self._scenario_metadata

    @scenario_metadata.setter
    def scenario_metadata(self, value):
        self._scenario_metadata = MetadataSet(value, self.regions, self.intervals)

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
        run_order = self._get_model_sets_in_run_order()
        timestep = self.timesteps[0]

        for model_set in run_order:
            model_set.run(timestep)

    def _run_sequential_sos_model(self):
        """Runs the system-of-system model sequentially
        """
        run_order = self._get_model_sets_in_run_order()
        self.logger.info("Determined run order as %s", run_order)
        for timestep in self.timesteps:
            for model_set in run_order:
                model_set.run(timestep)

    def run_sector_model(self, model_name):
        """Runs the sector model

        Parameters
        ----------
        model_name : str
            The name of the model, corresponding to the folder name in the
            models subfolder of the project folder
        """
        msg = "Model '{}' does not exist. Choose from {}"
        assert model_name in self.models, \
            msg.format(model_name, self.sector_models)

        msg = "Running the %s sector model"
        self.logger.info(msg, model_name)

        sector_model = self.models[model_name]

        # Run a simulation for a single model
        for timestep in self.timesteps:
            state, results = self.run_sector_model_timestep(sector_model, timestep)

            # Update state in next period to current post-decision state
            self.set_state(timestep, model_name, state)
            # Store the results for the current timestep
            self.set_data(sector_model, timestep, results)

    def run_sector_model_timestep(self, model, timestep):
        """Run the sector model for a specific timestep

        Parameters
        ----------
        model: :class:`smif.sector_model.SectorModel`
            The instance of the sector model wrapper to run
        timestep: int
            The year for which to run the model

        Returns
        -------
        state : list
            A list of :class:`smif.StateData`
        results : dict


        """
        self.logger.info("Running model %s for timestep %s",
                         model.name, timestep)
        state, decisions = self.get_state(timestep, model.name)
        data = self.get_data(model, timestep)

        state, results = model.simulate(decisions, state, data)
        self.logger.debug("Results from %s model:\n %s", model.name, results)
        return state, results

    def get_state(self, model_name, timestep):
        """Gets the state data and built interventions
        to pass to SectorModel.simulate

        Arguments
        ---------
        model_name : str
        timestep : int

        Returns
        -------
        state_data : list
            A list of :class:`smif.StateData`
        decisions : list

        """
        return self.state.get_all_state(model_name, timestep)

    def set_state(self, from_timestep, model_name, state):
        """Sets state output from model ready for next timestep

        Updates the state for the next timestep, if not in the final timestep

        Arguments
        ---------
        model_name : str
        timestep : int
        state : list
        """
        for_timestep = self.timestep_after(from_timestep)
        if for_timestep is not None:
            self.state.state_data = (for_timestep, model_name, state)

    def get_data(self, model, timestep):
        """Gets the data in the required format to pass to the simulate method

        Returns
        -------
        dict
            A nested dictionary of the format:
            ``data[parameter] = numpy.ndarray``

        Notes
        -----
        Note that the timestep is `not` passed to the SectorModel in the
        nested data dictionary.
        The current timestep is available in ``data['timestep']``.

        """
        new_data = {}
        timestep_idx = self.timesteps.index(timestep)

        for dependency in model.inputs.metadata:
            name = dependency.name
            provider = self.outputs[name]

            for source in provider:

                self.logger.debug("Getting '%s' dependency data for '%s' from '%s'",
                                  name, model.name, source)

                if source == 'scenario':
                    from_data = self.scenario_data[name][timestep_idx]
                    scenario_map = self.scenario_metadata
                    from_spatial_resolution = scenario_map.get_spatial_res(name)
                    from_temporal_resolution = scenario_map.get_temporal_res(name)
                    from_units = scenario_map.get_units(name)
                    self.logger.debug("Found data: %s", from_data)

                else:
                    source_model = self.models[source]
                    # get latest set of results from list
                    from_data = self.results[timestep][source][name]
                    from_spatial_resolution = source_model.outputs.get_spatial_res(name)
                    from_temporal_resolution = source_model.outputs.get_temporal_res(name)
                    from_units = source_model.outputs.get_units(name)
                    self.logger.debug("Found data: %s", from_data)

                to_spatial_resolution = dependency.spatial_resolution
                to_temporal_resolution = dependency.temporal_resolution
                to_units = dependency.units
                msg = "Converting from spatial resolution '%s' and  temporal resolution '%s'"
                self.logger.debug(msg, from_spatial_resolution, from_temporal_resolution)
                msg = "Converting to spatial resolution '%s' and  temporal resolution '%s'"
                self.logger.debug(msg, to_spatial_resolution, to_temporal_resolution)

                if from_units != to_units:
                    raise NotImplementedError("Units conversion not implemented %s - %s",
                                              from_units, to_units)

                if name not in new_data:
                    new_data[name] = self._convert_data(
                        from_data,
                        to_spatial_resolution,
                        to_temporal_resolution,
                        from_spatial_resolution,
                        from_temporal_resolution)
                else:
                    new_data[name] += self._convert_data(
                        from_data,
                        to_spatial_resolution,
                        to_temporal_resolution,
                        from_spatial_resolution,
                        from_temporal_resolution)

        new_data['timestep'] = timestep
        return new_data

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

    def _run_static_optimisation(self):
        """Runs the system-of-systems model in a static optimisation format
        """
        raise NotImplementedError

    def _run_dynamic_optimisation(self):
        """Runs the system-of-system models in a dynamic optimisation format
        """
        raise NotImplementedError

    def _get_model_sets_in_run_order(self):
        """Returns a list of :class:`ModelSet` in a runnable order.

        If a set contains more than one model, there is an interdependency and
        and we attempt to run the models to convergence.
        """
        if networkx.is_directed_acyclic_graph(self.dependency_graph):
            # topological sort gives a single list from directed graph, currently
            # ignoring opportunities to run independent models in parallel
            run_order = networkx.topological_sort(self.dependency_graph, reverse=True)

            # turn into a list of sets for consistency with the below
            ordered_sets = [
                ModelSet(
                    {self.models[model_name]},
                    self
                )
                for model_name in run_order
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
                        self.models[model_name]
                        for model_name in condensation.node[node_id]['members']
                    },
                    self
                )
                for node_id in networkx.topological_sort(condensation, reverse=True)
            ]

        return ordered_sets

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
            A list of timesteps, distinct and sorted in ascending order
        """
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value):
        self._timesteps = sorted(list(set(value)))

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
        return [intervention.name for intervention in self.state._interventions]

    @property
    def sector_models(self):
        """The list of sector model names

        Returns
        =======
        list
            A list of sector model names
        """
        return list(self.models.keys())

    @property
    def inputs(self):
        """Model names associated with inputs

        Returns
        -------
        dict
            Keys are parameter names, value is a list of sector model names
        """
        parameter_model_map = defaultdict(list)
        for model_name, model in self.models.items():
            for name in model.inputs.names:
                parameter_model_map[name].append(model_name)
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
        for model_name, model in self.models.items():
            for name in model.outputs.names:
                parameter_model_map[name].append(model_name)

        for name in self.scenario_metadata.names:
            parameter_model_map[name].append('scenario')
        return parameter_model_map


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
            state, results = self._sos_model.run_sector_model_timestep(model, timestep)
            self._sos_model.set_state(timestep,
                                      model.name,
                                      state)
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

                        state, results = self._sos_model.run_sector_model_timestep(
                            model, timestep)
                        self._sos_model.set_state(timestep,
                                                  model.name,
                                                  state)
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
            for output in model.outputs.metadata:
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
        param_names = [param.name for param in model.outputs.metadata]

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

        self.state_data = defaultdict(dict)
        self.interventions = InterventionRegister()
        self.planning = []

    def construct(self, config_data):
        """Set up the whole SosModel

        Parameters
        ----------
        config_data : dict
            A valid system-of-systems model configuration dictionary
        """
        model_list = config_data['sector_model_data']

        self.add_timesteps(config_data['timesteps'])
        self.set_max_iterations(config_data)
        self.set_convergence_abs_tolerance(config_data)
        self.set_convergence_rel_tolerance(config_data)

        self.load_region_sets(config_data['region_sets'])
        self.load_interval_sets(config_data['interval_sets'])

        self.add_planning(config_data['planning'])
        self.load_models(model_list)
        self.add_initial_state()

        self.add_scenario_metadata(config_data['scenario_metadata'])
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

    def add_scenario_metadata(self, scenario_metadata):
        """

        Parameters
        ----------
        scenario_metadata: list of dicts
            A dictionary containing information on the spatial and temporal
            resoultion of scenario data

        Example
        -------
        The data structure of each list item is as follows::

                [
                    {
                        'name': 'raininess',
                        'temporal_resolution': 'annual',
                        'spatial_resolution': 'LSOA',
                        'units': 'ml'
                    }
                ]

        """
        self.sos_model.scenario_metadata = scenario_metadata

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
            model = self._build_model(
                model_data,
                self.sos_model.regions,
                self.sos_model.intervals)
            self.add_model(model)
            self.add_model_data(model, model_data)

    @staticmethod
    def _build_model(model_data, regions, intervals):
        builder = SectorModelBuilder(model_data['name'])
        builder.load_model(model_data['path'], model_data['classname'])
        builder.create_initial_system(model_data['initial_conditions'])
        builder.add_regions(regions)
        builder.add_intervals(intervals)
        builder.add_inputs(model_data['inputs'])
        builder.add_outputs(model_data['outputs'])
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
        if model.regions is None:
            model.regions = self.sos_model.regions
        if model.intervals is None:
            model.intervals = self.sos_model.regions
        self.sos_model.models[model.name] = model

    def add_model_data(self, model, model_data):
        """Adds sector model data to the system-of-systems model which is
        convenient to have available at the higher level.
        """
        self._add_state_data(model.name, model_data['initial_conditions'])
        self._add_interventions(model.name, model_data['interventions'])

    def _add_interventions(self, model_name, interventions):
        """Adds interventions for a model to `SosModel` register
        """
        for intervention in interventions:
            intervention_object = Intervention(sector=model_name,
                                               data=intervention)
            msg = "Adding %s from %s to SosModel InterventionRegister"
            identifier = intervention_object.name
            self.logger.debug(msg, identifier, model_name)
            self.interventions.register(intervention_object)

    def _add_state_data(self, model_name, initial_conditions):
        """Add initial conditions to list of state data

        Assumes `is_state` values are associated with the first timeperiod

        Arguments
        ---------
        model_name : str
            The name of the model for which to add state data from initial
            conditions
        initial_conditions: list
            A list of past Interventions, with build dates and locations as
            necessary to specify the infrastructure system to be modelled.
        """
        # Collect model state data
        state_data = self._get_initial_conditions(initial_conditions)
        timestep = self.sos_model.timesteps[0]
        self.state_data[timestep][model_name] = state_data

    def _get_initial_conditions(self, initial_conditions):
        """Gets list of initial conditions

        Arguments
        ---------
        initial_conditions: list
            A list of past Interventions, with build dates and locations as
            necessary to specify the infrastructure system to be modelled.

        Returns
        -------
        list of :class:`smif.StateData`
        """
        state_data = filter(
            lambda d: len(d.data) > 0,
            [self._intervention_state_from_data(datum) for datum in initial_conditions]
        )
        return list(state_data)

    @staticmethod
    def _intervention_state_from_data(intervention_data):
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
        Planning register in the `state` attribute in the SosModel.

        Parameters
        ----------
        planning : list
            A list of planning instructions

        """
        self.logger.info("Adding planning")
        self.planning = Planning(planning)

    def add_initial_state(self):
        """Initialises the state class, and loads it into the SosModel

        Requires planning data, interventions and state data to be initialised
        """

        self.sos_model.state = State(self.planning,
                                     self.interventions)

        for timestep in self.state_data.keys():
            for model_name, data in self.state_data[timestep].items():
                self.sos_model.state.state_data = (timestep, model_name, data)

    def add_scenario_data(self, data):
        """Load the scenario data into the system of systems model

        Expect a dictionary, where each key maps a parameter
        name to a list of data, each observation with:

        - value
        - units
        - timestep (must use a timestep from the SoS model timesteps)
        - region (must use a region id from scenario regions)
        - interval (must use an id from scenario time intervals)

        Add a dictionary of :class:`numpy.ndarray`

                data[param] = np.zeros((num_timesteps, num_intervals, num_regions))
                data[param].fill(np.nan)
                # ...initially empty array then filled with data

        """
        self.logger.info("Adding scenario data")
        nested = {}

        for param, observations in data.items():
            if param not in self.sos_model.scenario_metadata.names:
                raise ValueError("Parameter {} not registered in scenario metadata {}".format(
                    param,
                    self.sos_model.scenario_metadata))
            param_metadata = self.sos_model.scenario_metadata[param]

            nested[param] = self._data_list_to_array(
                param,
                observations,
                self.sos_model.timesteps,
                param_metadata
            )

        self.sos_model._scenario_data = nested

    def _data_list_to_array(self, param, observations, timestep_names, param_metadata):
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
        interval_set_name = metadata.temporal_resolution
        interval_set = self.sos_model.intervals.get_intervals_in_set(interval_set_name)
        interval_names = [interval.name for key, interval in interval_set.items()]

        region_set_name = metadata.spatial_resolution
        region_set = self.sos_model.regions.get_regions_in_set(region_set_name)
        region_names = [region.name for region in region_set]

        if len(interval_names) == 0:
            self.logger.error("No interval names found when loading %s", param)

        if len(region_names) == 0:
            self.logger.error("No region names found when loading %s", param)

        return interval_names, region_names

    def _check_planning_interventions_exist(self):
        """Check existence of all the interventions in the pre-specifed planning

        """
        model = self.sos_model
        names = model.state._interventions.names
        for planning_name in model.state._planned.names:
            msg = "Intervention '{}' in planning file not found in interventions"
            assert planning_name in names, msg.format(planning_name)

    def _check_planning_timeperiods_exist(self):
        """Check existence of all the timeperiods in the pre-specified planning
        """
        model = self.sos_model
        model_timeperiods = model.timesteps
        for timeperiod in model.state._planned.timeperiods:
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
        available_intervals = self.sos_model.intervals.names
        msg = "Available time interval sets in SosModel: %s"
        self.logger.debug(msg, available_intervals)
        available_regions = self.sos_model.regions.names
        msg = "Available region sets in SosModel: %s"
        self.logger.debug(msg, available_regions)

        for model_name, model in self.sos_model.models.items():
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

        for model_name, model in self.sos_model.models.items():
            for dep in model.inputs.metadata:
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
                        dep_source = self.sos_model.scenario_metadata[dep.name]
                    else:
                        dep_source = self.sos_model.models[source]. \
                                     outputs[dep.name]
                    self.validate_dependency(dep_source, dep)

                    if source == 'scenario':
                        continue

                    dependency_graph.add_edge(model_name, source)

        self.sos_model.dependency_graph = dependency_graph

    def validate_dependency(self, source, sink):
        """For a source->sink pair of dependency metadata, validate viability
        of the conversion
        """
        if source.units != sink.units:
            raise AssertionError("Units %s, %s not compatible, conversion required by %s",
                                 source.units, sink.units, source.name)
        if source.spatial_resolution not in self.sos_model.regions.names:
            raise AssertionError("Region set %s not found, required by %s",
                                 source.spatial_resolution, source.name)
        if sink.spatial_resolution not in self.sos_model.regions.names:
            raise AssertionError("Region set %s not found, required by %s",
                                 sink.spatial_resolution, sink.name)
        if source.temporal_resolution not in self.sos_model.intervals.names:
            raise AssertionError("Interval set %s not found, required by %s",
                                 source.temporal_resolution, source.name)
        if sink.temporal_resolution not in self.sos_model.intervals.names:
            raise AssertionError("Interval set %s not found, required by %s",
                                 sink.temporal_resolution, sink.name)

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
