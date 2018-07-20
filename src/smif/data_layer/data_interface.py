"""Common data interface
"""
from abc import ABCMeta, abstractmethod
from logging import getLogger

import numpy as np


class DataInterface(metaclass=ABCMeta):
    """Abstract base class to define common data interface
    """
    def __init__(self):
        self.logger = getLogger(__name__)

    @abstractmethod
    def read_coefficients(self, source_name, destination_name):
        raise NotImplementedError

    @abstractmethod
    def write_coefficients(self, source_name, destination_name, data):
        raise NotImplementedError

    @abstractmethod
    def read_units_file_name(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model_runs(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model_run(self, sos_model_run_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model_run(self, sos_model_run_name, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def delete_sos_model_run(self, sos_model_run):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_models(self):
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model(self, sos_model):
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model(self, sos_model_name, sos_model):
        raise NotImplementedError()

    @abstractmethod
    def read_sector_models(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sector_model(self, sector_model_name):
        raise NotImplementedError()

    @abstractmethod
    def write_sector_model(self, sector_model):
        raise NotImplementedError()

    @abstractmethod
    def update_sector_model(self, sector_model_name, sector_model):
        raise NotImplementedError()

    @abstractmethod
    def read_region_definitions(self):
        raise NotImplementedError()

    @abstractmethod
    def read_region_definition_data(self, region_definition_name):
        raise NotImplementedError()

    @abstractmethod
    def write_region_definition(self, region_definition):
        raise NotImplementedError()

    @abstractmethod
    def update_region_definition(self, region_definition):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definitions(self):
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definition_data(self, interval_definition_name):
        raise NotImplementedError()

    @abstractmethod
    def write_interval_definition(self, interval_definition):
        raise NotImplementedError()

    @abstractmethod
    def update_interval_definition(self, interval_definition):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_set(self, scenario_set_name):
        raise NotImplementedError()

    @abstractmethod
    def read_strategies(self):
        raise NotImplementedError

    @abstractmethod
    def write_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario_set(self, scenario_set):
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_data(self, scenario_name, parameter_name,
                           spatial_resolution, temporal_resolution, timestep):
        raise NotImplementedError()

    @abstractmethod
    def write_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_sets(self):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_set(self, narrative_set_name):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def update_narrative_set(self, narrative_set):
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_data(self, narrative_name):
        raise NotImplementedError()

    @abstractmethod
    def write_narrative(self, narrative):
        raise NotImplementedError()

    @abstractmethod
    def update_narrative(self, narrative):
        raise NotImplementedError()

    @abstractmethod
    def read_state(self, modelrun_name, timestep=None, decision_iteration=None):
        """state is a list of (intervention_name, build_year), output of decision module/s
        """
        raise NotImplementedError()

    @abstractmethod
    def write_state(self, state, modelrun_name, timestep,
                    decision_iteration=None):
        """State is a list of decision dicts with name and build_year keys,

        State is output from the DecisionManager

        Arguments
        ---------
        state : list
        modelrun_name : str
        timestep: int
        decision_iteration : int, default=None
        """
        raise NotImplementedError()

    def read_parameters(self, modelrun_name, model_name):
        """Read global and model-specific parameter values for a given modelrun
        and model.
        """
        modelrun_config = self.read_sos_model_run(modelrun_name)
        params = {}
        for narratives in modelrun_config['narratives'].values():
            for narrative in narratives:
                data = self.read_narrative_data(narrative)
                for model_or_global, narrative_params in data.items():
                    if model_or_global in ('global', model_name):
                        params.update(narrative_params)
        return params

    @abstractmethod
    def read_results(self, modelrun_name, model_name, output_name, spatial_resolution,
                     temporal_resolution, timestep=None, modelset_iteration=None,
                     decision_iteration=None):
        raise NotImplementedError()

    @abstractmethod
    def write_results(self, modelrun_name, model_name, output_name, data, spatial_resolution,
                      temporal_resolution, timestep=None, modelset_iteration=None,
                      decision_iteration=None):
        raise NotImplementedError()

    @staticmethod
    def ndarray_to_data_list(data, region_names, interval_names, timestep=None):
        """Convert :class:`numpy.ndarray` to list of observations

        Parameters
        ----------
        data : numpy.ndarray
        region_names : list of str
        interval_names : list of str
        timestep: int or None

        Returns
        -------
        observations : list of dict
            Each dict has keys: 'region' (a region name), 'interval' (an
            interval name) and 'value'.
        """
        observations = []
        for region_idx, region in enumerate(region_names):
            for interval_idx, interval in enumerate(interval_names):
                observations.append({
                    'timestep': timestep,
                    'region': region,
                    'interval': interval,
                    'value': data[region_idx, interval_idx]
                })
        return observations

    @staticmethod
    def data_list_to_ndarray(observations, region_names, interval_names):
        """Convert list of observations to :class:`numpy.ndarray`

        Parameters
        ----------
        observations : list of dict
            Required keys for each dict are 'region' (a region name), 'interval'
            (an interval name) and 'value'.
        region_names : list
            A list of unique region names
        interval_names : list
            A list of unique interval names

        Returns
        -------
        data : numpy.ndarray

        Raises
        ------
        KeyError
            If an observation is missing a required key
        ValueError
            If an observation region or interval is not in region_names or
            interval_names
        DataNotFoundError
            If the observations don't include data for any region/interval
            combination
        DataMismatchError
            If the region_names and interval_names do not
            match the observations
        """
        # Check that the list of region and interval names are unique
        assert len(region_names) == len(set(region_names))
        assert len(interval_names) == len(set(interval_names))

        DataInterface._validate_observations(observations, region_names, interval_names)

        data = np.full((len(region_names), len(interval_names)), np.nan)

        for obs in observations:
            region_idx = region_names.index(obs['region'])
            interval_idx = interval_names.index(obs['interval'])
            data[region_idx, interval_idx] = obs['value']

        return data

    @staticmethod
    def _validate_observations(observations, region_names, interval_names):
        if len(observations) != len(region_names) * len(interval_names):
            msg = "Number of observations ({}) is not equal to intervals ({}) x regions ({})"
            raise DataMismatchError(
                msg.format(len(observations), len(region_names), len(interval_names))
            )
        DataInterface._validate_observation_keys(observations)
        DataInterface._validate_observation_meta(observations, region_names, 'region')
        DataInterface._validate_observation_meta(observations, interval_names, 'interval')

    @staticmethod
    def _validate_observation_keys(observations):
        for obs in observations:
            if 'region' not in obs:
                raise KeyError(
                    "Observation missing region: {}".format(obs))
            if 'interval' not in obs:
                raise KeyError(
                    "Observation missing interval: {}".format(obs))
            if 'value' not in obs:
                raise KeyError(
                    "Observation missing value: {}".format(obs))

    @staticmethod
    def _validate_observation_meta(observations, meta_list, meta_name):
        observed = set()
        for line, obs in enumerate(observations):
            if obs[meta_name] not in meta_list:
                raise ValueError("Unknown {} '{}' in row {}".format(
                    meta_name, obs[meta_name], line))
            else:
                observed.add(obs[meta_name])
        missing = set(meta_list) - observed
        if missing:
            raise DataNotFoundError(
                "Missing values for %ss: %s", meta_name, list(missing))


class DataNotFoundError(Exception):
    """Raise when some data is not found
    """
    pass


class DataExistsError(Exception):
    """Raise when some data is found unexpectedly
    """
    pass


class DataMismatchError(Exception):
    """Raise when some data doesn't match the context

    E.g. when updating an object by id, the updated object's id must match
    the id provided separately.
    """
    pass
