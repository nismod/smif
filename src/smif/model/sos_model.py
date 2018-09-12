# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

A system of systems model contains simulation and scenario models,
and the dependencies between the models.

"""
import logging

from smif.model.model import CompositeModel, Model
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
            'scenarios': sorted(
                scenario.name
                for scenario in self.scenario_models.values()
            ),
            'sector_models': sorted(
                model.name
                for model in self.sector_models.values()
            ),
            'scenario_dependencies': scenario_dependencies,
            'model_dependencies': model_dependencies
        }

        return config

    @classmethod
    def from_dict(cls, data, models=None):
        """Create a SosModel from config data

        Arguments
        ---------
        data: dict
            Configuration data. Must include name. May include description. If models are
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

        if isinstance(model, CompositeModel):
            msg = "Nesting of CompositeModels (including SosModels) is not supported"
            raise NotImplementedError(msg)

        self.logger.info("Loading model: %s", model.name)
        self.models[model.name] = model

    def before_model_run(self, data_handle):
        """Not implemented - use ModelRunner and a JobScheduler to execute
        """
        raise NotImplementedError("SosModel must be run by a scheduler")

    def simulate(self, data_handle):
        """Not implemented - use ModelRunner and a JobScheduler to execute
        """
        raise NotImplementedError("SosModel must be run by a scheduler")

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
