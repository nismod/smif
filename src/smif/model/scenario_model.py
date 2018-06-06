"""Scenario models represent scenario data sources within a system-of-systems
model.
"""
from logging import getLogger

from smif.model import Model


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
    timesteps : list
        List of timesteps for which the scenario holds data
    scenario_set : str
        Scenario set to which this scenario belongs
    """

    def __init__(self, name):
        super().__init__(name)
        self.scenario_set = None
        self.scenario_name = None

    def as_dict(self):
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

    def add_output(self, name, spatial_resolution, temporal_resolution, units):
        """Add an output to the scenario model

        Arguments
        ---------
        name: str
        spatial_resolution: :class:`smif.convert.area.RegionRegister`
        temporal_resolution: :class:`smif.convert.interval.TimeIntervalRegister`
        units: str

        """
        output_metadata = {
            "name": name,
            "spatial_resolution": spatial_resolution,
            "temporal_resolution": temporal_resolution,
            "units": units
        }
        self.outputs.add_metadata(output_metadata)

    def _check_output(self, output):
        if output not in self.outputs.names:
            raise KeyError("'{}' not in scenario outputs".format(output))

    def simulate(self, data):
        """No-op, as the data is assumed already available in the store
        """
        return data


class ScenarioModelBuilder(object):

    def __init__(self, name):
        self.scenario = ScenarioModel(name)
        self.logger = getLogger(__name__)

    def construct(self, scenario_config):
        """Build a ScenarioModel

        Arguments
        ---------
        scenario_config: dict
        """
        self.scenario.scenario_set = scenario_config['scenario_set']
        self.scenario.scenario_name = scenario_config['name']
        facets = scenario_config['facets']

        for facet in facets:
            spatial = facet['spatial_resolution']
            temporal = facet['temporal_resolution']

            spatial_res = self.scenario.regions.get_entry(spatial)
            temporal_res = self.scenario.intervals.get_entry(temporal)

            name = facet['name']
            self.scenario.add_output(name,
                                     spatial_res,
                                     temporal_res,
                                     facet['units'])

    def finish(self):
        """Return the built ScenarioModel
        """
        return self.scenario
