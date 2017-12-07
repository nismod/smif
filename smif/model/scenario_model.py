from logging import getLogger

import numpy as np
from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
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

        self._data = {}
        self.timesteps = []
        self.scenario_set = None

    def as_dict(self):
        config = {
            'name': self.name,
            'description': self.description,
            'scenario_set': self.scenario_set
        }

        parameters = [output.as_dict() for output in self.outputs.values()]
        config['parameters'] = parameters

        return config

    def get_data(self, output):
        """Get data associated with `output`

        Arguments
        ---------
        output : str
            The name of the output for which to retrieve data

        """
        self._check_output(output)
        return self._data[output]

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

    def add_data(self, output, data, timesteps):
        """Add data to the scenario

        Arguments
        ---------
        output : str
            The name of the output to which to add data
        data : numpy.ndarray
        timesteps : list

        Example
        -------
        >>> elec_scenario = ScenarioModel('elec_scenario')
        >>> data = np.array([[[120.23]]])
        >>> timesteps = [2010]
        >>> elec_scenario.add_data(data, timesteps)
        """
        self._check_output(output)

        self.timesteps = timesteps
        assert isinstance(data, np.ndarray)
        self._data[output] = data

    def _check_output(self, output):
        if output not in self.outputs.names:
            raise KeyError("'{}' not in scenario outputs".format(output))

    def simulate(self, timestep, data=None):
        """Returns the scenario data
        """
        time_index = self.timesteps.index(timestep)

        all_data = {
            output_name: self._data[output_name][time_index]
            for output_name in self.outputs
        }

        return {
            self.name: all_data
        }


class ScenarioModelBuilder(object):

    def __init__(self, name):
        self.scenario = ScenarioModel(name)
        self.logger = getLogger(__name__)

        self.region_register = get_region_register()
        self.interval_register = get_interval_register()

    def construct(self, scenario_config, data, timesteps):
        """Build the complete and populated ScenarioModel

        Assumes that a ScenarioModel can only have one output

        Arguments
        ---------
        scenario_config: dict
        data: dict
            A dictionary of scenario data, with keys scenario parameter names
        timesteps: list
            A list of integer years e.g. ``[2010, 2011, 2013]``
        """
        self.scenario.scenario_set = scenario_config['scenario_set']
        # Scenarios need to be known by the scenario set name
        self.scenario.name = scenario_config['scenario_set']
        parameters = scenario_config['parameters']

        for parameter in parameters:
            spatial = parameter['spatial_resolution']
            temporal = parameter['temporal_resolution']

            spatial_res = self.region_register.get_entry(spatial)
            temporal_res = self.interval_register.get_entry(temporal)

            name = parameter['name']
            self.scenario.add_output(name,
                                     spatial_res,
                                     temporal_res,
                                     parameter['units'])

            array_data = self._data_list_to_array(name,
                                                  data[name],
                                                  timesteps,
                                                  spatial_res,
                                                  temporal_res)

            self.scenario.add_data(name, array_data, timesteps)

    def finish(self):
        """Return the built ScenarioModel
        """
        return self.scenario

    def _data_list_to_array(self, param, observations, timestep_names,
                            spatial_resolution, temporal_resolution):
        # TODO push down into data_layer
        """Convert list of observations to :class:`numpy.ndarray`

        Arguments
        ---------
        param : str
        observations : list
        timestep_names : list
        spatial_resolution : smif.convert.area.RegionSet
        temporal_resolution : smif.convert.interval.IntervalSet

        """
        interval_names = temporal_resolution.get_entry_names()
        region_names = spatial_resolution.get_entry_names()

        self._validate_observations(param, observations, region_names, spatial_resolution.name,
                                    interval_names, temporal_resolution.name)

        if len(timestep_names) == 0:
            self.logger.error("No timesteps found when loading %s", param)

        data = np.full((
            len(timestep_names),
            len(region_names),
            len(interval_names)
        ), np.nan)

        if len(observations) != data.size:
            self.logger.warning(
                "Number of observations is not equal to timesteps x  " +
                "intervals x regions when loading %s", param)

        skipped_years = set()

        for obs in observations:
            year = int(obs['year'])
            region = obs['region']
            interval = obs['interval']

            if year not in timestep_names:
                # Don't add data if year is not in timestep list
                skipped_years.add(year)
                continue

            timestep_idx = timestep_names.index(year)
            interval_idx = interval_names.index(interval)
            region_idx = region_names.index(region)

            data[timestep_idx, region_idx, interval_idx] = obs['value']

        for year in skipped_years:
            msg = "Year '%s' not defined in model timesteps so skipping"
            self.logger.warning(msg, year)

        return data

    @staticmethod
    def _validate_observations(param, observations, region_names, region_set_name,
                               interval_names, interval_set_name):
        for obs in observations:
            if 'year' not in obs:
                raise ValueError(
                    "Scenario data item missing year: '{}'".format(obs))
            if 'region' not in obs:
                raise ValueError(
                    "Scenario data item missing region: '{}'".format(obs))
            if 'interval' not in obs:
                raise ValueError(
                    "Scenario data item missing interval: {}".format(obs))
            if 'value' not in obs:
                raise ValueError(
                    "Scenario data item missing value: {}".format(obs))

            if obs['region'] not in region_names:
                raise ValueError(
                    "Region '{}' not defined in set '{}' for parameter '{}'".format(
                        obs['region'],
                        region_set_name,
                        param))

            if obs['interval'] not in interval_names:
                raise ValueError(
                    "Interval '{}' not defined in set '{}' for parameter '{}'".format(
                        obs['interval'],
                        interval_set_name,
                        param))
