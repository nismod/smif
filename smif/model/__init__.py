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

>>> elec_scenario = ScenarioModel('scenario', ['demand'])
>>> elec_scenario.add_data({'demand': 123})
>>> sos_model = SosModel('simple')
>>> sos_model.add_model(elec_scenario)
>>> sos_model.simulate()
{'scenario': {'demand': 123}}

A more comprehensive example with one scenario and one scenario model:

>>>  elec_scenario = ScenarioModel('scenario', ['output'])
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

from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.metadata import MetadataSet
from smif.model.composite import Dependency


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

        self.regions = get_region_register()
        self.intervals = get_interval_register()

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
        self.logger.debug("Dependencies: %s", self.deps)
        free_input_names = model_inputs - set(self.deps.keys())

        return MetadataSet([self._model_inputs[name]
                           for name in free_input_names])

    @abstractmethod
    def simulate(self, timestep, data=None):
        pass

    def add_dependency(self, source_model, source, sink, function=None):
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
            source_object = source_model.model_outputs[source]
            self.deps[sink] = (Dependency(source_model,
                                          source_object,
                                          function))
            msg = "Added dependency from '%s' to '%s'"
            self.logger.debug(msg, source_model.name, self.name)
        else:
            if sink in self.model_inputs.names:
                raise NotImplementedError("Multiple source dependencies"
                                          " not yet implemented")

            msg = "Inputs: '%s'. Free inputs: '%s'."
            self.logger.debug(msg, self.model_inputs.names,
                              self.free_inputs.names)
            msg = "Input '{}' is not defined in '{}' model"
            raise ValueError(msg.format(sink, self.name))


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
