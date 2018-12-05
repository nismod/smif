# -*- coding: utf-8 -*-
"""This module coordinates the software components that make up the integration
framework.

A system of systems model contains simulation and scenario models,
and the dependencies between the models.

"""
import itertools
import logging
from collections import defaultdict

from smif.exception import SmifDataMismatchError, SmifValidationError
from smif.metadata import RelativeTimestep
from smif.model.dependency import Dependency
from smif.model.model import Model, ScenarioModel
from smif.model.sector_model import SectorModel

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class SosModel():
    """Consists of the collection of models joined via dependencies

    Arguments
    ---------
    name : str
        The unique name of the SosModel

    """
    def __init__(self, name):
        self.logger = logging.getLogger(__name__)

        self.name = name
        self.description = ''

        # Models in an internal lookup by name
        # { name: Model }
        self._models = {}

        # Maintain dependencies in an internal lookup (using names (str) in the tuple key)
        # { (source, output, sink, input): list[Dependency]}
        self._scenario_dependencies = defaultdict(list)
        self._model_dependencies = defaultdict(list)

        self.narratives = {}

    def as_dict(self):
        """Serialize the SosModel object

        Returns
        -------
        dict
        """
        scenario_dependencies = []
        for dep in self._scenario_dependencies.values():
            scenario_dependencies.append(dep.as_dict())

        model_dependencies = []
        for dep in self._model_dependencies.values():
            model_dependencies.append(dep.as_dict())

        config = {
            'name': self.name,
            'description': self.description,
            'scenarios': sorted(
                scenario.name
                for scenario in self.scenario_models
            ),
            'sector_models': sorted(
                model.name
                for model in self.sector_models
            ),
            'scenario_dependencies': scenario_dependencies,
            'model_dependencies': model_dependencies,
            'narratives': self.narratives
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

        try:
            sos_model.description = data['description']
        except KeyError:
            pass

        if models:
            for model in models:
                sos_model.add_model(model)

            for dep in data['model_dependencies'] + data['scenario_dependencies']:
                sink = SosModel._get_dependency(sos_model, dep, 'sink')
                source = SosModel._get_dependency(sos_model, dep, 'source')
                source_output_name = dep['source_output']
                sink_input_name = dep['sink_input']
                try:
                    timestep = RelativeTimestep.from_name(dep['timestep'])
                except KeyError:
                    # if not provided, default to current
                    timestep = RelativeTimestep.CURRENT
                except ValueError:
                    # if not parseable as relative timestep, pass through
                    pass

                sos_model.add_dependency(
                    source, source_output_name, sink, sink_input_name, timestep)

        sos_model.check_dependencies()
        return sos_model

    @staticmethod
    def _get_dependency(sos_model, dep, source_or_sink):
        try:
            source = sos_model.get_model(dep[source_or_sink])
        except KeyError:
            msg = 'SectorModel or ScenarioModel {} `{}` required by ' + \
                  'dependency `{}` was not provided by the builder'
            dependency = (
                dep['source'] + ' (' + dep['source_output'] + ')' +
                ' - ' +
                dep['sink'] + ' (' + dep['sink_input'] + ')'
            )
            raise SmifDataMismatchError(msg.format(source_or_sink,
                                                   dep[source_or_sink],
                                                   dependency))
        return source

    @property
    def models(self):
        """The models contained within this system-of-systems model

        Returns
        -------
        list[smif.model.model.Model]
        """
        return list(self._models.values())

    @property
    def sector_models(self):
        """Sector model objects contained in the SosModel

        Returns
        =======
        list
            A list of sector model objects
        """
        return [
            model for model in self.models
            if isinstance(model, SectorModel)
        ]

    @property
    def scenario_models(self):
        """Scenario model objects contained in the SosModel

        Returns
        -------
        list
            A list of scenario model objects
        """
        return [
            model for model in self.models
            if isinstance(model, ScenarioModel)
        ]

    def add_model(self, model):
        """Adds a model to the system-of-systems model

        Arguments
        ---------
        model : :class:`smif.model.model.Model`
        """
        msg = "Only Models can be added to a SosModel (and SosModels cannot be nested)"
        assert isinstance(model, Model), msg
        self.logger.info("Loading model: %s", model.name)
        self._models[model.name] = model

    def get_model(self, model_name):
        """Get a model by name

        Arguments
        ---------
        model_name : str

        Returns
        -------
        smif.model.model.Model
        """
        return self._models[model_name]

    @property
    def dependencies(self):
        """Dependency connections between models within this system-of-systems model

        Returns
        -------
        iterable[smif.model.dependency.Dependency]
        """
        return itertools.chain(
            self._scenario_dependencies.values(),
            self._model_dependencies.values()
        )

    @property
    def scenario_dependencies(self):
        """Dependency connections between models within this system-of-systems model

        Returns
        -------
        iterable[smif.model.dependency.Dependency]
        """
        return self._scenario_dependencies.values()

    @property
    def model_dependencies(self):
        """Dependency connections between models within this system-of-systems model

        Returns
        -------
        iterable[smif.model.dependency.Dependency]
        """
        return self._model_dependencies.values()

    def add_dependency(self, source_model, source_output_name, sink_model, sink_input_name,
                       timestep=RelativeTimestep.CURRENT):
        """Adds a dependency to this system-of-systems model

        Arguments
        ---------
        source_model : smif.model.model.Model
            A reference to the source `~smif.model.model.Model` object
        source_output_name : string
            The name of the model_output defined in the `source_model`
        source_model : smif.model.model.Model
            A reference to the source `~smif.model.model.Model` object
        sink_name : string
            The name of a model_input defined in this object
        timestep : smif.metadata.RelativeTimestep, optional
            The relative timestep of the dependency, defaults to CURRENT, may be PREVIOUS.
        """
        try:
            self.get_model(source_model.name)
        except KeyError:
            msg = "Source model '{}' does not exist in list of models"
            raise SmifValidationError(msg.format(source_model.name))

        try:
            self.get_model(sink_model.name)
        except KeyError:
            msg = "Sink model '{}' does not exist in list of models"
            raise SmifValidationError(msg.format(sink_model.name))

        if source_output_name not in source_model.outputs:
            msg = "Output '{}' is not defined in '{}' model"
            raise SmifValidationError(msg.format(source_output_name, source_model.name))

        if sink_input_name not in sink_model.inputs:
            msg = "Input '{}' is not defined in '{}' model"
            raise SmifValidationError(msg.format(sink_input_name, sink_model.name))

        key = (source_model.name, source_output_name, sink_model.name, sink_input_name)

        if not self._allow_adding_dependency(
                source_model, source_output_name, sink_model, sink_input_name, timestep):
            msg = "Inputs: '%s'. Free inputs: '%s'."
            self.logger.debug(msg, sink_model.inputs, self.free_inputs)
            msg = "Could not add dependency: input '{}' already provided"
            raise SmifValidationError(msg.format(sink_input_name))

        source_spec = source_model.outputs[source_output_name]
        sink_spec = sink_model.inputs[sink_input_name]
        if isinstance(source_model, ScenarioModel):
            self._scenario_dependencies[key] = Dependency(
                source_model, source_spec, sink_model, sink_spec, timestep=timestep)
        else:
            self._model_dependencies[key] = Dependency(
                source_model, source_spec, sink_model, sink_spec, timestep=timestep)

        msg = "Added dependency from '%s:%s' to '%s:%s'"
        self.logger.debug(
            msg, source_model.name, source_output_name, self.name, sink_input_name)

    def _allow_adding_dependency(self, source_model, source_output_name, sink_model,
                                 sink_input_name, timestep):
        key = (source_model.name, source_output_name, sink_model.name, sink_input_name)
        existing_deps = key in self._scenario_dependencies or key in self._model_dependencies

        if existing_deps:
            if key in self._scenario_dependencies and key in self._model_dependencies:
                # allowed at most two sources
                allowable = False
            else:
                # if and only if one is a scenario and the other is to a previous timestep
                try:
                    other_dep = self._scenario_dependencies[key]
                except KeyError:
                    other_dep = self._model_dependencies[key]

                other_model = other_dep.source_model
                other_timestep = other_dep.timestep

                other_is_previous = other_timestep == RelativeTimestep.PREVIOUS
                other_is_scenario = isinstance(other_model, ScenarioModel)

                new_is_previous = timestep == RelativeTimestep.PREVIOUS
                new_is_scenario = isinstance(source_model, ScenarioModel)

                allowable = (new_is_previous and other_is_scenario) or \
                    (other_is_previous and new_is_scenario)
        else:
            allowable = True

        return allowable

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

    @property
    def free_inputs(self):
        """Returns the free inputs to models which are not yet linked to a dependency.

        Returns
        -------
        dict of {(model_name, input_name): smif.metadata.Spec}
        """
        satisfied_inputs = set(
            (sink_name, sink_input_name)
            for (source_name, source_output_name, sink_name, sink_input_name), deps
            in itertools.chain(
                self._scenario_dependencies.items(), self._model_dependencies.items())
            if deps
        )
        all_inputs = set()
        all_specs = {}

        for model in self.models:
            try:
                for input_name, input_spec in model.inputs.items():
                    input_key = (model.name, input_name)
                    all_inputs.add(input_key)
                    all_specs[input_key] = input_spec
            except AttributeError:
                pass  # ScenarioModels don't have inputs

        unsatisfied_inputs = all_inputs - satisfied_inputs

        return {
            input_key: all_specs[input_key]
            for input_key in unsatisfied_inputs
        }

    @property
    def outputs(self):
        """All model outputs provided by models contained within this SosModel

        Returns
        -------
        dict of {(model_name, output_name): smif.metadata.Spec}
        """
        outputs = {}
        for model in self.models:
            for output_name, output_spec in model.outputs.items():
                outputs[(model.name, output_name)] = output_spec

        return outputs

    def add_narrative(self, narrative):
        """Add a narrative to the system-of-systems model

        Arguments
        ---------
        narrative: dict
        """
        try:
            name = narrative['name']
        except KeyError:
            msg = "Could not find narrative name. Narrative argument " \
                  "should be a dict. Received a '{}'."
            raise SmifDataMismatchError(msg.format(type(narrative)))

        for model, parameters in narrative['provides'].items():
            self._check_model_exists(model)
            for parameter in parameters:
                self._check_parameter_exists(parameter, model)

        self.narratives[name] = narrative

    def _check_model_exists(self, model_name):
        if model_name not in [model.name for model in self.models]:
            msg = "'{}' does not exist in '{}'"
            raise SmifDataMismatchError(msg.format(model_name, self.name))

    def _check_parameter_exists(self, parameter_name, model_name):
        model = self._models[model_name]
        if parameter_name not in model.parameters:
            msg = "Parameter '{}' does not exist in '{}'"
            raise SmifDataMismatchError(msg.format(parameter_name, model_name))
