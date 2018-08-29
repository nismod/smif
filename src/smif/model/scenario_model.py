"""Scenario models represent scenario data sources within a system-of-systems model.
"""
from smif.metadata import Spec
from smif.model.model import Model


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

    def add_output(self, spec):
        """Add an output to the scenario model

        Arguments
        ---------
        config: dict
        """
        self.outputs[spec.name] = spec

    def simulate(self, data):
        """No-op, as the data is assumed to be already available in the store
        """
        return data
