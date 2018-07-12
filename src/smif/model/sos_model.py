# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

A system of systems model contains simulation and scenario models,
and the dependencies between the models.

"""
import logging

import networkx
from smif.model import CompositeModel, Model, element_after, element_before
from smif.model.model_set import ModelSet
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SosModel(CompositeModel):
    """Consists of the collection of models joined via dependencies

    This class is populated at runtime by the :class:`SosModelBuilder` and
    called from :func:`smif.cli.run_model`.  SosModel inherits from
    :class:`smif.composite.Model`.

    Arguments
    ---------
    name : str
        The unique name of the SosModel

    """
    def __init__(self, name):
        # housekeeping
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self.max_iterations = 25
        self.convergence_relative_tolerance = 1e-05
        self.convergence_absolute_tolerance = 1e-08

        # models - includes types of SectorModel and ScenarioModel
        self.dependency_graph = networkx.DiGraph()

        self.timesteps = []

    def as_dict(self):
        """Serialize the SosModel object

        Returns
        -------
        dict
        """

        dependencies = []
        for model in self.models.values():
            for name, dep in model.deps.items():
                dep_config = {'source_model': dep.source_model.name,
                              'source_model_output': dep.source.name,
                              'sink_model': model.name,
                              'sink_model_input': name}
                dependencies.append(dep_config)

        config = {
            'name': self.name,
            'description': self.description,
            'scenario_sets': [scenario.scenario_set
                              for scenario in self.scenario_models.values()],
            'sector_models': list(self.sector_models.keys()),
            'dependencies': dependencies,
            'max_iterations': self.max_iterations,
            'convergence_absolute_tolerance': self.convergence_absolute_tolerance,
            'convergence_relative_tolerance': self.convergence_relative_tolerance
        }

        return config

    def add_model(self, model):
        """Adds a sector model to the system-of-systems model

        Arguments
        ---------
        model : :class:`smif.sector_model.SectorModel`
            A sector model wrapper

        """
        assert isinstance(model, Model)
        self.logger.info("Loading model: %s", model.name)
        self.models[model.name] = model

    def before_model_run(self, data_handle):
        """Initialise each model (passing in parameter data only)
        """
        for model in self.sector_models.values():
            # get custom data handle for the Model
            model_data_handle = data_handle.derive_for(model)
            model.before_model_run(model_data_handle)

    def simulate(self, data_handle):
        """Run the SosModel

        Arguments
        ---------
        data_handle: smif.data_layer.DataHandle
            Access state, parameter values, dependency inputs

        Returns
        -------
        results : smif.data_layer.DataHandle
            Access model outputs

        """
        self.make_dependency_graph()
        run_order = self._get_model_sets_in_run_order()
        self.logger.info("Determined run order as %s", [x.name for x in run_order])
        for model in run_order:
            self.logger.info("*** Running the %s model ***", model.name)
            # Pass simulate access to a DataHandle derived for the particular
            # model
            model.simulate(data_handle.derive_for(model))
        return data_handle

    def make_dependency_graph(self):
        """For each contained model, compare dependency list against
        list of available models and build the dependency graph
        """
        for model in self.models.values():
            self.dependency_graph.add_node(model, name=model.name)

        for sink_model in self.models.values():
            for dependency in sink_model.deps.values():
                source_model = dependency.source_model
                self.dependency_graph.add_edge(source_model, sink_model)

    def _get_model_sets_in_run_order(self):
        """Returns a list of :class:`Model` in a runnable order.

        If a set contains more than one model, there is an interdependency and
        and we attempt to run the models to convergence.

        Returns
        -------
        list
            A list of `smif.model.Model` objects
        """
        if networkx.is_directed_acyclic_graph(self.dependency_graph):
            # topological sort gives a single list from directed graph, currently
            # ignoring opportunities to run independent models in parallel
            run_order = networkx.topological_sort(self.dependency_graph)

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
            for node_id in networkx.topological_sort(condensation):
                models = condensation.node[node_id]['members']
                if len(models) == 1:
                    ordered_sets.append(models.pop())
                else:
                    ordered_sets.append(ModelSet(
                        {model.name: model for model in models},
                        max_iterations=self.max_iterations,
                        relative_tolerance=self.convergence_relative_tolerance,
                        absolute_tolerance=self.convergence_absolute_tolerance))

        return ordered_sets

    def timestep_before(self, timestep):
        """Returns the timestep previous to a given timestep, or None

        Arguments
        ---------
        timestep : str

        Returns
        -------
        str

        """
        return element_before(timestep, self.timesteps)

    def timestep_after(self, timestep):
        """Returns the timestep after a given timestep, or None

        Arguments
        ---------
        timestep : str

        Returns
        -------
        str
        """
        return element_after(timestep, self.timesteps)


    @property
    def sector_models(self):
        """Sector model objects contained in the SosModel

        Returns
        =======
        dict
            A dict of sector model objects
        """
        return {x: y for x, y in self.models.items()
                if isinstance(y, SectorModel)}

    @property
    def scenario_models(self):
        """Scenario model objects contained in the SosModel

        Returns
        -------
        dict
            A dict of scenario model objects
        """
        return {x: y for x, y in self.models.items()
                if isinstance(y, ScenarioModel)}


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
    def __init__(self, name='global'):
        self.sos_model = SosModel(name)
        self.logger = logging.getLogger(__name__)

    def construct(self, sos_model_config):
        """Set up the whole SosModel

        Parameters
        ----------
        sos_model_config : dict
            A valid system-of-systems model configuration dictionary
        """
        self.sos_model.name = sos_model_config['name']
        self.sos_model.description = sos_model_config['description']
        self.set_max_iterations(sos_model_config)
        self.set_convergence_abs_tolerance(sos_model_config)
        self.set_convergence_rel_tolerance(sos_model_config)

        self.load_models(sos_model_config['sector_models'])
        self.load_scenario_models(sos_model_config['scenario_sets'])

        self.add_dependencies(sos_model_config['dependencies'])

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
        self.logger.debug("Available models: %s", self.sos_model.models.keys())
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

    def load_models(self, model_list):
        """Loads the sector models into the system-of-systems model

        Parameters
        ----------
        model_list : list
            A list of SectorModel objects
        """
        self.logger.info("Loading models")
        for model in model_list:
            self.sos_model.add_model(model)

    def load_scenario_models(self, scenario_list):
        """Loads the scenario models into the system-of-systems model

        Parameters
        ----------
        scenario_list : list
            A list of ScenarioModel objects

        """
        self.logger.info("Loading scenarios")
        for scenario in scenario_list:
            self.sos_model.add_model(scenario)

    def check_dependencies(self):
        """For each contained model, compare dependency list against
        list of available models
        """
        if self.sos_model.free_inputs.names:
            msg = "A SosModel must have all inputs linked to dependencies. " \
                  "Define dependencies for {}"
            raise NotImplementedError(
                msg.format(", ".join(self.sos_model.free_inputs.names)))

        for model in self.sos_model.models.values():
            if isinstance(model, SosModel):
                msg = "Nesting of SosModels not yet supported"
                raise NotImplementedError(msg)

        for sink_model in self.sos_model.models.values():
            for dependency in sink_model.deps.values():
                self._validate_each_dependency(dependency)

    def _validate_each_dependency(self, dependency):
        source_model = dependency.source_model
        msg = "Dependency '%s' provided by '%s'"
        self.logger.debug(msg, dependency.sink.name, source_model.name)

        # Insist on identical metadata - conversions to be explicit
        if dependency.source.spatial_resolution.name != \
                dependency.sink.spatial_resolution.name:
            self.logger.warn(
                "Implicit spatial conversion attempted ({}>{})".format(
                    dependency.source.spatial_resolution.name,
                    dependency.sink.spatial_resolution.name))

        if dependency.source.temporal_resolution.name != \
                dependency.sink.temporal_resolution.name:
            self.logger.warn(
                "Implicit temporal conversion attempted ({}>{})".format(
                    dependency.source.temporal_resolution.name,
                    dependency.sink.temporal_resolution.name))

        if dependency.source.units != dependency.sink.units:
            self.logger.warn(
                "Implicit units conversion (%s>%s)",
                dependency.source.units,
                dependency.sink.units)

        source_units = dependency.source.units
        sink_units = dependency.sink.units

        if source_model.units.parse_unit(source_units) is None:
            msg = "Cannot convert from undefined unit '{}'"
            raise ValueError(msg.format(source_units))
        if source_model.units.parse_unit(sink_units) is None:
            msg = "Cannot convert to undefined unit '{}'"
            raise ValueError(msg.format(sink_units))

        if source_units != sink_units:
            source_model.units.get_coefficients(source_units,
                                                sink_units)

    def finish(self):
        """Returns a configured system-of-systems model ready for operation

        Includes validation steps, e.g. to check dependencies
        """
        self.check_dependencies()
        return self.sos_model
