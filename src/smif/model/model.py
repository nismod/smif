"""Model abstract class
"""
import sys
from abc import ABCMeta, abstractmethod
from enum import Enum
from logging import getLogger

from smif.metadata import Spec


class ModelOperation(Enum):
    """ Enumerate that describes the possible operations on Models
    """
    BEFORE_MODEL_RUN = 'before_model_run'
    SIMULATE = 'simulate'


class Model(metaclass=ABCMeta):
    """Abstract class represents the interface used to implement the model classes
    `SectorModel` and `ScenarioModel`.

    Arguments
    ---------
    name : str
    """

    def __init__(self, name):
        self.logger = getLogger(__name__)

        self.name = name
        self.description = ''
        self._inputs = {}
        self._parameters = {}
        self._outputs = {}

    def __repr__(self):
        return "<{} name='{}'>".format(self.__class__.__name__, self.name)

    @classmethod
    def from_dict(cls, config):
        """Create object from dictionary serialisation
        """
        model = cls(config['name'])
        model.description = config['description']
        for input_ in config['inputs']:
            model.add_input(Spec.from_dict(input_))
        for output in config['outputs']:
            model.add_output(Spec.from_dict(output))
        for param in config['parameters']:
            model.add_parameter(Spec.from_dict(param))
        return model

    def as_dict(self):
        """Serialize the SectorModel object as a dictionary

        Returns
        -------
        dict
        """
        config = {
            'name': self.name,
            'description': self.description,
            'path': sys.modules[self.__module__].__file__,
            'classname': self.__class__.__name__,
            'inputs': [inp.as_dict() for inp in self.inputs.values()],
            'outputs': [out.as_dict() for out in self.outputs.values()],
            'parameters': [param.as_dict() for param in self.parameters.values()]
        }
        return config

    @property
    def inputs(self):
        """All model inputs defined at this layer

        Returns
        -------
        dict of {input_name: smif.metadata.Spec}
        """
        return self._inputs

    @property
    def parameters(self):
        """Model parameters

        Returns
        -------
        dict of {parameter_name: smif.data_layer.data_array.DataArray}
        """
        return self._parameters

    def add_input(self, spec):
        """Add an input

        Arguments
        ---------
        spec: smif.metadata.Spec
        """
        self.inputs[spec.name] = spec

    def add_parameter(self, spec):
        """Add a parameter

        Arguments
        ---------
        spec: smif.metadata.Spec
        """
        self.parameters[spec.name] = spec

    @property
    def outputs(self):
        """All model outputs defined at this layer

        Returns
        -------
        dict of {output_name: smif.metadata.Spec}
        """
        return self._outputs

    def add_output(self, spec):
        """Add an output

        Arguments
        ---------
        spec: smif.metadata.Spec
        """
        self.outputs[spec.name] = spec

    @abstractmethod
    def simulate(self, data):
        """Override to implement the generation of model results

        Generate ``results`` for ``timestep`` using ``data``

        Arguments
        ---------
        data: smif.data_layer.DataHandle
            Access state, parameter values, dependency inputs.
        """


class ScenarioModel(Model):
    """Represents exogenous scenario data

    Arguments
    ---------
    name : str
        The name of this scenario (scenario set/abstract
        scenario/scenario group) - like sector model name

    Attributes
    ----------
    name : str
        Name of this scenario
    scenario : str
        Instance of scenario (concrete instance)
    """
    def __init__(self, name):
        super().__init__(name)
        self.scenario = None

    @classmethod
    def from_dict(cls, data):
        """Create ScenarioModel from dict serialisation
        """
        scenario = cls(data['name'])
        scenario.scenario = data['scenario']
        if 'description' in data:
            scenario.description = data['description']
        for output in data['outputs']:
            spec = Spec.from_dict(output)
            scenario.add_output(spec)

        return scenario

    def as_dict(self):
        """Serialise ScenarioModel to dict
        """
        config = {
            'name': self.name,
            'description': self.description,
            'scenario': self.scenario,
            'outputs': [
                output.as_dict()
                for output in self.outputs.values()
            ]
        }
        return config

    def simulate(self, data):
        """No-op, as the data is assumed to be already available in the store
        """
        return data
