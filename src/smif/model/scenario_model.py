"""Scenario models represent scenario data sources within a system-of-systems model.
"""
from smif.metadata import Spec
from smif.model.model import Model


class ScenarioModel(Model):
    """Represents exogenous scenario data

    Arguments
    ---------
    name : str
        The unique name of this scenario

    Attributes
    ----------
    name : str
        Name of this scenario
    scenario_set : str
        Scenario set to which this scenario belongs
    scenario_name : str
        Scenario represented
    """
    def __init__(self, name):
        super().__init__(name)
        self.scenario_set = None
        self.scenario_name = None

    @classmethod
    def from_dict(cls, data):
        """Create ScenarioModel from dict serialisation
        """
        scenario = cls(data['name'])
        scenario.scenario_set = data['scenario_set']
        scenario.scenario_name = data['scenario_name']
        for facet in data['facets']:
            spec = Spec.from_dict(facet)
            scenario.add_output(spec)

        return scenario

    def as_dict(self):
        """Serialise ScenarioModel to dict
        """
        config = {
            'name': self.name,
            'description': self.description,
            'scenario_set': self.scenario_set,
            'facets': [
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
