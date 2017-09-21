from logging import getLogger

import numpy as np
from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.metadata import MetadataSet
from smif.model import Model


class ScenarioModel(Model):
    """Represents exogenous scenario data

    Arguments
    ---------
    name : string
        The unique name of this scenario
    output : smif.metadata.MetaData
        A name for the scenario output parameter
    """

    def __init__(self, name, output=None):
        if output:
            if isinstance(output, MetadataSet):
                super().__init__(name)
                self._model_outputs = output
            else:
                msg = "output argument should be type smif.metadata.MetadataSet"
                raise TypeError(msg)
        else:
            super().__init__(name)

        self._data = {}
        self.timesteps = []

    @property
    def data(self):
        return self._data

    def add_output(self, name, spatial_resolution, temporal_resolution, units):
        """Add an output to the scenario model

        Arguments
        ---------
        name: str
        spatial_resolution: :class:`smif.convert.area.RegionRegister`
        temporal_resolution: :class:`smif.convert.interval.TimeIntervalRegister`
        units: str

        """
        output_metadata = {"name": name,
                           "spatial_resolution": spatial_resolution,
                           "temporal_resolution": temporal_resolution,
                           "units": units}

        self._model_outputs.add_metadata(output_metadata)

    def add_data(self, data, timesteps):
        """Add data to the scenario

        Arguments
        ---------
        data : numpy.ndarray
        timesteps : list

        Example
        -------
        >>> elec_scenario = ScenarioModel('elec_scenario')
        >>> data = np.array([[[120.23]]])
        >>> timesteps = [2010]
        >>> elec_scenario.add_data(data, timesteps)
        """
        self.timesteps = timesteps
        assert isinstance(data, np.ndarray)
        self._data = data

    def simulate(self, timestep, data=None):
        """Returns the scenario data
        """
        time_index = self.timesteps.index(timestep)
        return {self.name: {self.model_outputs.names[0]: self._data[time_index]}}


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
        scenario_config : dict
        data : list
        timesteps : list
        """
        spatial = scenario_config['spatial_resolution']
        temporal = scenario_config['temporal_resolution']

        spatial_res = self.region_register.get_entry(spatial)
        temporal_res = self.interval_register.get_entry(temporal)

        self.scenario.scenario_set = scenario_config['scenario_set']
        name = scenario_config['name']
        self.scenario.name = name
        self.scenario.add_output(scenario_config['parameter'],
                                 spatial_res,
                                 temporal_res,
                                 scenario_config['units'])

        self.scenario.filename = scenario_config['filename']

        array_data = self._data_list_to_array(name, data,
                                              timesteps,
                                              spatial_res,
                                              temporal_res)

        self.scenario.add_data(array_data, timesteps)

    def finish(self):
        """Return the built ScenarioModel
        """
        return self.scenario

    def _data_list_to_array(self, param, observations, timestep_names,
                            spatial_resolution, temporal_resolution):
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

        if len(timestep_names) == 0:
            self.logger.error("No timesteps found when loading %s", param)

        data = np.zeros((
            len(timestep_names),
            len(region_names),
            len(interval_names)
        ))
        data.fill(np.nan)

        if len(observations) != data.size:
            self.logger.warning(
                "Number of observations is not equal to timesteps x  " +
                "intervals x regions when loading %s", param)

        skipped_years = set()

        for obs in observations:

            if 'year' not in obs:
                raise ValueError(
                    "Scenario data item missing year: '{}'".format(obs))
            year = obs['year']

            if year not in timestep_names:
                # Don't add data if year is not in timestep list
                skipped_years.add(year)
                continue

            if 'region' not in obs:
                raise ValueError(
                    "Scenario data item missing region: '{}'".format(obs))
            region = obs['region']
            if region not in region_names:
                raise ValueError(
                    "Region '{}' not defined in set '{}' for parameter '{}'".format(
                        region,
                        spatial_resolution.name,
                        param))

            if 'interval' not in obs:
                raise ValueError(
                    "Scenario data item missing interval: {}".format(obs))
            interval = obs['interval']
            if interval not in interval_names:
                raise ValueError(
                    "Interval '{}' not defined in set '{}' for parameter '{}'".format(
                        interval,
                        temporal_resolution.name,
                        param))

            timestep_idx = timestep_names.index(year)
            interval_idx = interval_names.index(interval)
            region_idx = region_names.index(region)

            data[timestep_idx, region_idx, interval_idx] = obs['value']

        for year in skipped_years:
            msg = "Year '%s' not defined in model timesteps so skipping"
            self.logger.warning(msg, year)

        return data
