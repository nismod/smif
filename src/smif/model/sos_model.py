# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

A system of systems model contains simulation and scenario models,
and the dependencies between the models.

"""
import logging

import networkx
from smif.model.model import CompositeModel, Model
from smif.model.model_set import ModelSet
from smif.model.scenario_model import ScenarioModel
from smif.model.sector_model import SectorModel

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SosModel(CompositeModel):
    """Consists of the collection of models joined via dependencies

    SosModel inherits from :class:`smif.model.CompositeModel`.

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
        self.dependency_graph = None

    def as_dict(self):
        """Serialize the SosModel object

        Returns
        -------
        dict
        """
        scenario_dependencies = []
        model_dependencies = []
        for model in self.models.values():
            for dep in model.deps.values():
                dep_config = {
                    'source': dep.source_model.name,
                    'source_output': dep.source.name,
                    'sink': dep.sink_model.name,
                    'sink_input': dep.sink.name
                }
                if isinstance(dep.source_model, ScenarioModel):
                    scenario_dependencies.append(dep_config)
                else:
                    model_dependencies.append(dep_config)

        config = {
            'name': self.name,
            'description': self.description,
            'scenario_sets': [
                scenario.name
                for scenario in self.scenario_models.values()
            ],
            'sector_models': [
                model.name
                for model in self.sector_models.values()
            ],
            'scenario_dependencies': scenario_dependencies,
            'model_dependencies': model_dependencies,
            'max_iterations': self.max_iterations,
            'convergence_absolute_tolerance': self.convergence_absolute_tolerance,
            'convergence_relative_tolerance': self.convergence_relative_tolerance
        }

        return config

    @classmethod
    def from_dict(cls, data, models=None):
        """Create a SosModel from config data

        Arguments
        ---------
        data: dict
            Configuration data. Must include name. May include description, max_iterations,
            convergence_absolute_tolerance, convergence_relative_tolerance. If models are
            provided, must include dependencies.
        models: list of Model
            Optional. If provided, must include each ScenarioModel and SectorModel referred to
            in the data['dependencies']
        """
        sos_model = cls(data['name'])

        def test(key, dict_):
            """Quick existence-and-not-None check
            """
            return key in dict_ and dict_[key] is not None

        if test('description', data):
            sos_model.description = data['description']

        # convergence settings - eventually hoist to model runner with dataflow implementation
        if test('max_iterations', data):
            sos_model.max_iterations = data['max_iterations']
        if test('convergence_absolute_tolerance', data):
            sos_model.convergence_absolute_tolerance = data['convergence_absolute_tolerance']
        if test('convergence_relative_tolerance', data):
            sos_model.convergence_relative_tolerance = data['convergence_relative_tolerance']

        # models
        if models:
            for model in models:
                sos_model.add_model(model)

            for dep in _collect_dependencies(data):
                sink = sos_model.models[dep['sink']]
                source = sos_model.models[dep['source']]
                source_output_name = dep['source_output']
                sink_input_name = dep['sink_input']

                sink.add_dependency(source, source_output_name, sink_input_name)
            sos_model.check_dependencies()
        return sos_model

    def add_model(self, model):
        """Adds a sector model to the system-of-systems model

        Arguments
        ---------
        model : :class:`smif.sector_model.Model`
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
        run_order = self._get_model_sets_in_run_order()
        self.logger.info("Determined run order as %s", [x.name for x in run_order])
        for model in run_order:
            self.logger.info("*** Running the %s model ***", model.name)
            # Pass simulate access to a DataHandle derived for the particular
            # model
            model.simulate(data_handle.derive_for(model))
        return data_handle

    @staticmethod
    def make_dependency_graph(models):
        """Build a networkx DiGraph from models (as nodes) and dependencies (as edges)
        """
        dependency_graph = networkx.DiGraph()
        for model in models.values():
            dependency_graph.add_node(model, name=model.name)

        for model in models.values():
            for dependency in model.deps.values():
                dependency_graph.add_edge(
                    dependency.source_model,
                    dependency.sink_model
                )
        return dependency_graph

    def _get_model_sets_in_run_order(self):
        """Returns a list of :class:`Model` in a runnable order.

        If a set contains more than one model, there is an interdependency and
        and we attempt to run the models to convergence.

        Returns
        -------
        list
            A list of `smif.model.Model` objects
        """
        graph = SosModel.make_dependency_graph(self.models)
        if networkx.is_directed_acyclic_graph(graph):
            # topological sort gives a single list from directed graph, currently
            # ignoring opportunities to run independent models in parallel
            run_order = networkx.topological_sort(graph)

            # list of Models (typically ScenarioModel and SectorModel)
            ordered_sets = list(run_order)

        else:
            # contract the strongly connected components (subgraphs which
            # contain cycles) into single nodes, producing the 'condensation'
            # of the graph, where each node maps to one or more sector models
            condensation = networkx.condensation(graph)

            # topological sort of the condensation gives an ordering of the
            # contracted nodes, whose 'members' attribute refers back to the
            # original dependency graph
            ordered_sets = []
            for node_id in networkx.topological_sort(condensation):
                models = condensation.node[node_id]['members']
                if len(models) == 1:
                    ordered_sets.append(models.pop())
                else:
                    ordered_sets.append(
                        ModelSet(
                            {model.name: model for model in models},
                            max_iterations=self.max_iterations,
                            relative_tolerance=self.convergence_relative_tolerance,
                            absolute_tolerance=self.convergence_absolute_tolerance
                        )
                    )

        return ordered_sets

    @property
    def sector_models(self):
        """Sector model objects contained in the SosModel

        Returns
        =======
        dict
            A dict of sector model objects
        """
        return {
            name: model for name, model in self.models.items()
            if isinstance(model, SectorModel)
        }

    @property
    def scenario_models(self):
        """Scenario model objects contained in the SosModel

        Returns
        -------
        dict
            A dict of scenario model objects
        """
        return {
            name: model for name, model in self.models.items()
            if isinstance(model, ScenarioModel)
        }

    def check_dependencies(self):
        """For each contained model, compare dependency list against
        list of available models
        """
        if self.free_inputs:
            msg = "A SosModel must have all inputs linked to dependencies. " \
                  "Define dependencies for {}"
            raise NotImplementedError(msg.format(", ".join(
                str(key) for key in self.free_inputs.keys()
            )))

        for model in self.models.values():
            if isinstance(model, CompositeModel):
                msg = "Nesting of CompositeModels (including SosModels) is not supported"
                raise NotImplementedError(msg)


def _collect_dependencies(data):
    """Return all dependencies from a SosModel config dict
    """
    deps = []
    if 'dependencies' in data:
        deps = deps + data['dependencies']
    if 'model_dependencies' in data:
        deps = deps + data['model_dependencies']
    if 'scenario_dependencies' in data:
        deps = deps + data['scenario_dependencies']
    return deps
