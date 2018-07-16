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

>>> elec_scenario = ScenarioModel('scenario')
>>> elec_scenario.add_output('demand', 'national', 'annual', 'GWh')
>>> sos_model = SosModel('simple')
>>> sos_model.add_model(elec_scenario)
>>> sos_model.simulate(2010)
{'scenario': {'demand': array([123])}}

A more comprehensive example with one scenario and one scenario model:

>>> elec_scenario = ScenarioModel('scenario')
>>> elec_scenario.add_output('demand', 'national', 'annual', 'GWh')
>>> class EnergyModel(SectorModel):
...   def extract_obj(self):
...     pass
...   def simulate(self, timestep, data):
...     return {self.name: {'cost': data['input'] * 2}}
...
>>> energy_model = EnergyModel('model')
>>> energy_model.add_input('input', 'national', 'annual', 'GWh')
>>> energy_model.add_dependency(elec_scenario, 'demand', 'input', lambda x: x)
>>> sos_model = SosModel('sos')
>>> sos_model.add_model(elec_scenario)
>>> sos_model.add_model(energy_model)
>>> sos_model.simulate(2010)
{'model': {'cost': array([[246]])}, 'scenario': {'demand': array([[123]])}}

"""
from abc import ABCMeta, abstractmethod
from logging import getLogger

from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.convert.unit import get_register as get_unit_register
from smif.metadata import MetadataSet
from smif.model.dependency import Dependency
from smif.parameters import ParameterList


class Model(metaclass=ABCMeta):
    """Abstract class represents the interface used to implement the composite
    `SosModel` and leaf classes `SectorModel` and `Scenario`.

    Arguments
    ---------
    name : str
    inputs : smif.metadata.MetaDataSet
    outputs : smif.metadata.MetaDataSet

    """
    regions = get_region_register()
    intervals = get_interval_register()
    units = get_unit_register()

    def __init__(self, name):
        self.name = name
        self.description = ''
        self._inputs = MetadataSet([])
        self._outputs = MetadataSet([])
        self.deps = {}

        self._parameters = ParameterList()

        self.timesteps = []

        self.logger = getLogger(__name__)

    @property
    def inputs(self):
        """All model inputs defined at this layer

        Returns
        -------
        smif.metadata.MetadataSet
        """
        return self._inputs

    @property
    def outputs(self):
        """All model outputs defined at this layer

        Returns
        -------
        smif.metadata.MetadataSet

        """
        return self._outputs

    @property
    def free_inputs(self):
        """Returns the free inputs not linked to a dependency at this layer

        Free inputs are passed up to higher layers for deferred linkages to
        dependencies.

        Returns
        -------
        smif.metadata.MetadataSet
        """
        all_input_names = set(self.inputs.names)
        dep_input_names = set(dep.sink.name for dep in self.deps.values())
        free_input_names = all_input_names - dep_input_names

        return MetadataSet(self.inputs[name] for name in free_input_names)

    @abstractmethod
    def simulate(self, data):
        """Override to implement the generation of model results

        Generate ``results`` for ``timestep`` using ``data``

        Arguments
        ---------
        data: smif.data_layer.DataHandle
            Access state, parameter values, dependency inputs.
        """
        pass

    def add_dependency(self, source_model, source_name, sink_name, function=None):
        """Adds a dependency to the current `Model` object

        Arguments
        ---------
        source_model : `smif.composite.Model`
            A reference to the source `~smif.composite.Model` object
        source_name : string
            The name of the model_output defined in the `source_model`
        sink_name : string
            The name of a model_input defined in this object

        """
        if source_name not in source_model.outputs.names:
            msg = "Output '{}' is not defined in '{}' model"
            raise ValueError(msg.format(source_name, source_model.name))

        if sink_name in self.free_inputs.names:
            source = source_model.outputs[source_name]
            sink = self.inputs[sink_name]
            self.deps[sink_name] = Dependency(
                source_model,
                source,
                sink,
                function
            )
            msg = "Added dependency from '%s:%s' to '%s:%s'"
            self.logger.debug(msg, source_model.name, source_name, self.name, sink_name)

        else:
            if sink_name in self.inputs.names:
                raise NotImplementedError("Multiple source dependencies"
                                          " not yet implemented")

            msg = "Inputs: '%s'. Free inputs: '%s'."
            self.logger.debug(msg, self.inputs.names, self.free_inputs.names)
            msg = "Input '{}' is not defined in '{}' model"
            raise ValueError(msg.format(sink_name, self.name))

    def add_parameter(self, parameter_dict):
        """Add a parameter to the model

        Arguments
        ---------
        parameter_dict : dict
            Contains the keys ``name``, ``description``,  ``absolute_range``,
            ``suggested_range``, ``default_value``, ``units``
        """
        self._parameters.add_parameter(parameter_dict)

    @property
    def parameters(self):
        """A list of parameters

        Returns
        -------
        smif.parameters.ParameterList
        """
        return self._parameters


class CompositeModel(Model, metaclass=ABCMeta):
    """Override to implement models which contain models.

    Inherited by `smif.model.sos_model.SosModel` and
    `smif.model.model_set.ModelSet`
    """

    def __init__(self, name):
        super().__init__(name)
        self.models = {}

    @property
    def free_inputs(self):
        """Returns the free inputs not linked to a dependency at this layer

        For this composite :class:`~smif.model.CompositeModel` this includes
        the free_inputs from all contained smif.model.Model objects

        Free inputs are passed up to higher layers for deferred linkages to
        dependencies.

        Returns
        -------
        smif.metadata.MetadataSet
        """
        # free inputs of current layer
        free_inputs = super().free_inputs.metadata

        # free inputs of all contained models
        for model in self.models.values():
            free_inputs.extend(model.free_inputs.metadata)

        # compose a new MetadataSet containing the free inputs
        metadataset = MetadataSet(free_inputs)

        return metadataset

    @property
    def outputs(self):
        outputs = super().outputs.metadata
        for model in self.models.values():
            outputs.extend(model.outputs.metadata)

        return MetadataSet(outputs)


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
