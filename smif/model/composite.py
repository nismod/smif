"""Implements a composite scenario/sector model/system-of-systems model

Begin by declaring the atomic units (scenarios and sector models) which make up
a the composite system-of-systems model and then add these to the composite.
Declare dependencies by using the ``add_dependency()`` method, passing in a
reference to the source model object, and a pointer to the model output
and sink parameter name for the destination model.

Run the model by calling the ``simulate()`` method, passing in a dictionary
containing data for any free hanging model inputs, not linked through a
dependency. A fully defined SosModel should have no hanging model inputs, and
can therefore be called using ``simulate()`` with no arguments.

Responsibility for passing required data to the contained models lies with the
calling class. This means data is only ever passed one layer down.
This simplifies the interface, and allows as little or as much hiding of data,
dependencies and model inputs as required.

Example
-------
A very simple example with just one scenario:

>>> elec_scenario = Scenario('scenario', ['demand'])
>>> elec_scenario.add_data({'demand': 123})
>>> sos_model = SosModel('simple')
>>> sos_model.add_model(elec_scenario)
>>> sos_model.simulate()
{'scenario': {'demand': 123}}

A more comprehensive example with one scenario and one scenario model:

>>>  elec_scenario = Scenario('scenario', ['output'])
>>>  elec_scenario.add_data({'output': 123})
>>>  energy_model = SectorModel('model', [], [])
>>>  energy_model.add_input('input')
>>>  energy_model.add_dependency(elec_scenario, 'output', 'input')
>>>  energy_model.add_executable(lambda x: x)
>>>  sos_model = SosModel('blobby')
>>>  sos_model.add_model(elec_scenario)
>>>  sos_model.add_model(energy_model)
>>>  sos_model.simulate()
{'model': {'input': 123}, 'scenario': {'output': 123}}

"""

from abc import ABC, abstractmethod
from logging import getLogger

from smif.metadata import MetadataSet


class Model(ABC):
    """Abstract class represents the interface used to implement the composite
    `SosModel` and leaf classes `SectorModel` and `Scenario`.

    Arguments
    ---------
    name : str
    inputs : smif.metadata.MetaDataSet
    outputs : smif.metadata.MetaDataSet

    """

    def __init__(self, name):
        self.name = name
        self._model_inputs = MetadataSet([])
        self._model_outputs = MetadataSet([])
        self.deps = {}

        self.logger = getLogger(__name__)

    @property
    def model_inputs(self):
        """All model inputs defined at this layer

        Returns
        -------
        smif.metadata.MetadataSet
        """
        return self._model_inputs

    @property
    def model_outputs(self):
        """All model outputs defined at this layer

        Returns
        -------
        smif.metadata.MetadataSet

        """
        return self._model_outputs

    @property
    def free_inputs(self):
        """Returns the free inputs not linked to a dependency at this layer

        Free inputs are passed up to higher layers for deferred linkages to
        dependencies.

        Returns
        -------
        smif.metadata.MetadataSet
        """
        if self._model_inputs.names:
            model_inputs = set(self._model_inputs.names)
        else:
            model_inputs = set()

        self.logger.debug("Model inputs to %s: %s", self.name, model_inputs)
        self.logger.debug("Dependencies: %s", list(self.deps.keys()))
        free_input_names = model_inputs - set(self.deps.keys())

        return MetadataSet([self._model_inputs[name]
                           for name in free_input_names])

    @abstractmethod
    def simulate(self, timestep, data=None):
        pass

    def add_dependency(self, source_model, source, sink):
        """Adds a dependency to the current `Model` object

        Arguments
        ---------
        source_model : `smif.composite.Model`
            A reference to the source `~smif.composite.Model` object
        source : string
            The name of the model_output defined in the `source_model`
        sink : string
            The name of a model_input defined in this object

        """
        if source not in source_model.model_outputs.names:
            msg = "Output '{}' is not defined in '{}' model"
            raise ValueError(msg.format(source, source_model.name))
        if sink in self.free_inputs.names:
            self.deps[sink] = Dependency(source_model, source)
            msg = "Added dependency from '%s' to '%s'"
            self.logger.debug(msg, source_model, self.name)
        else:
            msg = "Input '{}' is not defined in '{}' model"
            raise ValueError(msg.format(sink, self.name))


class ScenarioModel(Model):
    """Represents exogenous scenario data

    Arguments
    ---------
    name : string
        The unique name of this scenario
    output : list
        A name for the scenario output parameter
    """

    def __init__(self, name, output):
        assert len(output) == 1
        super().__init__(name, [], output)
        self._data = []

    def add_data(self, data):
        """Add data to the scenario

        Arguments
        ---------
        data : dict
            Key of dict should be name which matches output name
        """
        self._data = data

    def simulate(self, timestep, data=None):
        """Returns the scenario data
        """
        return self._data[timestep]


class SectorModel(Model):
    """A sector model object represents a simulation model
    """

    def __init__(self, name, model_inputs, model_outputs):
        super().__init__(name, model_inputs, model_outputs)
        self._executable = None

    def add_input(self, input_name):
        self._model_inputs.append(input_name)

    def add_output(self, output_name):
        self._model_outputs.append(output_name)

    def add_executable(self, executable):
        """The function run when the simulate method is called

        ``executable`` is the wrapper around the sector model
        and must accept a dictionary of model inputs which matches
        the list of ``SectorModel.model_inputs``
        """
        self._executable = executable

    def simulate(self, timestep, data=None):
        """Simulates the sector model

        Arguments
        ---------
        input_data : dict
        """
        self.logger.debug("Running %s with data: %s", self.name, data)
        return self._executable(timestep, data)


class SosModel(Model):
    """A system-of-systems container object.

    A container for other subclasses of ``Model`` including ``SectorModel``,
    ``Scenario`` and ``SosModel``.

    Arguments
    ---------
    name : string
        The unique identifier for this system-of-systems model container
    """

    def __init__(self, name):
        super().__init__(name, [], [])
        self._models = {}

    def add_model(self, model):
        """Add a child model to the system-of-systems container
        """
        self._model_inputs.extend(model.free_inputs)
        self._model_outputs.extend(model.model_outputs)
        self._models[model.name] = model
        self.logger.debug("Added model '%s' to SosModel", model.name)

    def simulate(self, timestep, data=None):
        """Run the simulation for this and any contained ``Model`` objects

        Arguments
        ---------
        data : dict
            Data passed in to meet a dependency from an upper layer
        """
        self.logger.debug('%s Running Simulate', self.name)
        self.logger.debug('%s passed into %s.simulate()', data, self.name)

        results = {}

        # Replace this with code to detect loops in graph of submodels in this
        # layer
        for model_name, model in self._models.items():
            for model_input in model.model_inputs:
                if not data or model_input not in data:
                    if not data:
                        data = {}
                    if model_input not in model.deps:
                        msg = "Dependency not found for '{}'"
                        raise ValueError(msg.format(model_input))
                    else:
                        dependency = model.deps[model_input]
                        self.logger.debug("Found dependency for '%s'",
                                          model_input)
                        data[model_input] = dependency.get_data()

            results[model_name] = model.simulate(timestep, data)

        return results


class Dependency():
    """

    Arguments
    ---------
    source_model : smif.composite.Model
        The source model object
    source : string
        The name of the source parameter (output)
    function=None : func
        A conversion function
    """

    def __init__(self, source_model, source, function=None):
        self.source_model = source_model
        self.source = source
        if function:
            self._function = function
        else:
            self._function = self.convert

    @staticmethod
    def convert(data):
        return data

    def get_data(self, timestep):
        data = self.source_model.simulate(timestep)
        return self._function(data[self.source])

    def __repr__(self):
        return "Dependency('{}', '{}')".format(self.source_model,
                                               self.source)
