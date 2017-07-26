"""The Model Run collects scenarios, timesteps, narratives, and
model collection into a package which can be built and passed to
the ModelRunner to run.

The ModelRunner is responsible for running a ModelRun, including passing
in the correct data to the model between timesteps and calling to the
DecisionManager to obtain decisions.

ModeRun has attributes:
- id
- description
- sosmodel
- timesteps
- scenarios
- narratives
- strategy
- status

"""
from logging import getLogger

import numpy as np
from smif.convert.area import RegionRegister, RegionSet
from smif.convert.interval import TimeIntervalRegister
from smif.metadata import MetadataSet
from smif.sos_model import SosModelBuilder


class ModelRun(object):
    """
    """

    def __init__(self):

        self._name = 0
        self.description = ""
        self.sos_model = None
        self._model_horizon = []
        self.scenarios = {}
        self._scenario_metadata = MetadataSet({})
        self._scenario_data = {}
        self.narratives = None
        self.strategy = None
        self.status = 'Empty'

        self.logger = getLogger(__name__)

        # space and time
        self.regions = RegionRegister()
        self.intervals = TimeIntervalRegister()

    @property
    def name(self):
        """Unique identifier of the ModelRun
        """
        return self._name

    @property
    def scenario_metadata(self):
        """Returns the temporal and spatial mapping to an input, output or scenario parameter
        """
        return self._scenario_metadata

    @scenario_metadata.setter
    def scenario_metadata(self, value):
        self._scenario_metadata = MetadataSet(value, self.regions, self.intervals)

    @property
    def scenario_data(self):
        """Get nested dict of scenario data

        Returns
        -------
        dict
            Nested dictionary in the format ``data[year][param] =
            SpaceTimeValue(region, interval, value, unit)``
        """
        return self._scenario_data

    @property
    def model_horizon(self):
        """Returns the list of timesteps

        Returns
        =======
        list
            A list of timesteps, distinct and sorted in ascending order
        """
        return self._model_horizon

    @model_horizon.setter
    def model_horizon(self, value):
        self._model_horizon = sorted(list(set(value)))

    def run(self):
        """Builds all the objects and passes them to the ModelRunner

        The idea is that this will add ModelRuns to a queue for asychronous
        processing

        """
        self.logger.debug("Running model run %s", self.name)
        if self.status == 'Built':
            self.status = 'Running'
            modelrunner = ModelRunner()
            modelrunner.solve_model(self)
            self.status = 'Successful'
        else:
            raise ValueError("Model is not yet built.")


class ModelRunner(object):
    """Runs a ModelRun
    """

    def __init__(self):
        self.logger = getLogger(__name__)

    def solve_model(self, model_run):
        """Solve a ModelRun

        Arguments
        ---------
        model_run : :class:`smif.modelrun.ModelRun`
        """
        for timestep in model_run.model_horizon:
            self.logger.debug('Running model for timestep %s', timestep)
            model_run.sos_model.run(timestep)


class ModelRunBuilder(object):
    """Builds the ModelRun object from the configuration
    """
    def __init__(self):
        self.model_run = ModelRun()
        self.logger = getLogger(__name__)

    def construct(self, config_data):
        """Set up the whole ModelRun

        Parameters
        ----------
        config_data : dict
            A valid system-of-systems model configuration dictionary
        """
        self._add_timesteps(config_data['timesteps'])
        self._add_sos_model(config_data)

        self.load_region_sets(config_data['region_sets'])
        self.load_interval_sets(config_data['interval_sets'])

        self._add_scenario_metadata(config_data['scenario_metadata'])
        self._add_scenario_data(config_data['scenario_data'],
                                config_data['timesteps'])

    def finish(self):
        """Returns a configured model run ready for operation

        """
        return self.model_run

    def _add_sos_model(self, config_data):
        """
        """
        builder = SosModelBuilder()
        builder.construct(config_data, self.model_run.model_horizon)
        self.model_run.sos_model = builder.finish()

    def _add_timesteps(self, timesteps):
        """Set the timesteps of the system-of-systems model

        Parameters
        ----------
        timesteps : list
            A list of timesteps
        """
        self.logger.info("Adding timesteps to model run")
        self.model_run.model_horizon = timesteps

    def load_region_sets(self, region_sets):
        """Loads the region sets into the system-of-system model

        Parameters
        ----------
        region_sets: list
            A dict, where key is the name of the region set, and the value
            the data
        """
        assert isinstance(region_sets, dict)

        region_set_definitions = region_sets.items()
        if len(region_set_definitions) == 0:
            msg = "No region sets have been defined"
            self.logger.warning(msg)
        for name, data in region_set_definitions:
            msg = "Region set data is not a list"
            assert isinstance(data, list), msg
            self.model_run.regions.register(RegionSet(name, data))

    def load_interval_sets(self, interval_sets):
        """Loads the time-interval sets into the system-of-system model

        Parameters
        ----------
        interval_sets: list
            A dict, where key is the name of the interval set, and the value
            the data
        """
        interval_set_definitions = interval_sets.items()
        if len(interval_set_definitions) == 0:
            msg = "No interval sets have been defined"
            self.logger.warning(msg)

        for name, data in interval_set_definitions:
            self.model_run.intervals.register(data, name)

    def _add_scenario_metadata(self, scenario_metadata):
        """

        Parameters
        ----------
        scenario_metadata: list of dicts
            A dictionary containing information on the spatial and temporal
            resoultion of scenario data

        Example
        -------
        The data structure of each list item is as follows::

                [
                    {
                        'name': 'raininess',
                        'temporal_resolution': 'annual',
                        'spatial_resolution': 'LSOA',
                        'units': 'ml'
                    }
                ]

        """
        self.model_run.scenario_metadata = scenario_metadata

    def _add_scenario_data(self, data, timesteps):
            """Load the scenario data into the system of systems model

            Expect a dictionary, where each key maps a parameter
            name to a list of data, each observation with:

            - value
            - units
            - timestep (must use a timestep from the SoS model timesteps)
            - region (must use a region id from scenario regions)
            - interval (must use an id from scenario time intervals)

            Add a dictionary of :class:`numpy.ndarray`

                    data[param] = np.zeros((num_timesteps, num_intervals, num_regions))
                    data[param].fill(np.nan)
                    # ...initially empty array then filled with data

            """
            self.logger.info("Adding scenario data")
            nested = {}

            for param, observations in data.items():
                if param not in self.model_run.scenario_metadata.names:
                    msg = "Parameter {} not registered in scenario metadata {}"
                    raise ValueError(msg.format(
                        param,
                        self.model_run.scenario_metadata))
                param_metadata = self.model_run.scenario_metadata[param]

                nested[param] = self._data_list_to_array(
                    param,
                    observations,
                    timesteps,
                    param_metadata
                )

            self.model_run.scenarios = nested

    def _data_list_to_array(self, param, observations, timestep_names,
                            param_metadata):
        """Convert list of observations to :class:`numpy.ndarray`
        """
        interval_names, region_names = self._get_dimension_names_for_param(
            param_metadata, param)

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

        for obs in observations:
            if 'year' not in obs:
                raise ValueError("Scenario data item missing year: {}".format(obs))
            year = obs['year']
            if year not in timestep_names:
                raise ValueError(
                    "Year {} not defined in model timesteps".format(year))

            if 'region' not in obs:
                raise ValueError("Scenario data item missing region: {}".format(obs))
            region = obs['region']
            if region not in region_names:
                raise ValueError(
                    "Region {} not defined in set {} for parameter {}".format(
                        region,
                        param_metadata.spatial_resolution,
                        param))

            if 'interval' not in obs:
                raise ValueError("Scenario data item missing interval: {}".format(obs))
            interval = obs['interval']
            if interval not in interval_names:
                raise ValueError(
                    "Interval {} not defined in set {} for parameter {}".format(
                        interval,
                        param_metadata.temporal_resolution,
                        param))

            timestep_idx = timestep_names.index(year)
            interval_idx = interval_names.index(interval)
            region_idx = region_names.index(region)

            data[timestep_idx, region_idx, interval_idx] = obs['value']

        return data

    def _get_dimension_names_for_param(self, metadata, param):
        interval_set_name = metadata.temporal_resolution
        interval_set = self.model_run.intervals.get_intervals_in_set(interval_set_name)
        interval_names = [interval.name for key, interval in interval_set.items()]

        region_set_name = metadata.spatial_resolution
        region_set = self.model_run.regions.get_regions_in_set(region_set_name)
        region_names = [region.name for region in region_set]

        if len(interval_names) == 0:
            self.logger.error("No interval names found when loading %s", param)

        if len(region_names) == 0:
            self.logger.error("No region names found when loading %s", param)

        return interval_names, region_names
