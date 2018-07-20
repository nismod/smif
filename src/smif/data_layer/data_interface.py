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
    def read_units_file_name(self):
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model_runs(self):
        """Read all system-of-system model runs

        Returns
        -------
        list
            A list of sos_model_run dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model_run(self, sos_model_run_name):
        """Read a system-of-system model run

        Arguments
        ---------
        sos_model_run_name: str
            A sos_model_run name

        Returns
        -------
        sos_model_run: dict
            A sos_model_run dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def write_sos_model_run(self, sos_model_run):
        """Write system-of-system model run

        Arguments
        ---------
        sos_model_run: dict
            A sos_model_run dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model_run(self, sos_model_run_name, sos_model_run):
        """Update system-of-system model run

        Arguments
        ---------
        sos_model_run_name: str
            A sos_model_run name
        sos_model_run: dict
            A sos_model_run dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_sos_model_run(self, sos_model_run):
        """Delete a system-of-system model run

        Arguments
        ---------
        sos_model_run_name: str
            A sos_model_run name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_sos_models(self):
        """Read all system-of-system models

        Returns
        -------
        list
            A list of sos_models dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_sos_model(self, sos_model_name):
        """Read a specific system-of-system model

        Arguments
        ---------
        sos_model_name: str
            A sos_model name

        Returns
        -------
        sos_model: dict
            A sos_model dictionary
        """

    @abstractmethod
    def write_sos_model(self, sos_model):
        """Write system-of-system model

        Arguments
        ---------
        sos_model: dict
            A sos_model dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def update_sos_model(self, sos_model_name, sos_model):
        """Update system-of-system model

        Arguments
        ---------
        sos_model_name: str
            A sos_model name
        sos_model: dict
            A sos_model dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_sos_model(self, sos_model_name):
        """Delete a system-of-system model

        Arguments
        ---------
        sos_model_name: str
            A sos_model name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_sector_models(self):
        """Read all sector models

        sector_models.yml

        Returns
        -------
        list
            A list of sector_model dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_sector_model(self, sector_model_name):
        """Read a sector model

        Arguments
        ---------
        sector_model_name: str
            A sector_model name

        Returns
        -------
        sector_model: dict
            A sector_model dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def write_sector_model(self, sector_model):
        """Write sector model

        Arguments
        ---------
        sector_model: dict
            A sector_model dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def update_sector_model(self, sector_model_name, sector_model):
        """Update sector model

        Arguments
        ---------
        sector_model_name: str
            A sector_model name
        sector_model: dict
            A sector_model dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_sector_model(self, sector_model_name):
        """Delete a sector model

        Arguments
        ---------
        sector_model_name: str
            A sector_model name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        """Read list of (name, build_year) for a given modelrun, timestep,
        decision
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

    @abstractmethod
    def read_region_definitions(self):
        """Read region_definitions from project configuration

        Returns
        -------
        list
            A list of region_definition dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_region_definition_data(self, region_definition_name):
        """Read region_definition data file into a Fiona feature collection

        The file format must be possible to parse with GDAL, and must contain
        an attribute "name" to use as an identifier for the region_definition.

        Arguments
        ---------
        region_definition_name: str
            Name of the region_definition

        Returns
        -------
        list
            A list of data from the specified file in a fiona formatted dict
        """
        raise NotImplementedError()

    @abstractmethod
    def write_region_definition(self, region_definition):
        """Write region_definition to project configuration

        Arguments
        ---------
        region_definition: dict
            A region_definition dict

        Notes
        -----
        Unused
        """
        raise NotImplementedError()

    @abstractmethod
    def update_region_definition(self, region_definition):
        """Update region_definition to project configuration

        Arguments
        ---------
        region_definition_name: str
            Name of the (original) entry
        region_definition: dict
            The updated region_definition dict

        Notes
        -----
        Unused
        """
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definitions(self):
        """Read interval_definition sets from project configuration

        Returns
        -------
        list
            A list of interval_definition set dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_interval_definition_data(self, interval_definition_name):
        """Read data for an interval definition

        Arguments
        ---------
        interval_definition_name: str

        Returns
        -------
        dict
            Interval definition data

        Notes
        -----
        Expects headings of `id`, `start`, `end`
        """
        raise NotImplementedError()

    @abstractmethod
    def write_interval_definition(self, interval_definition):
        """Write interval_definition to project configuration

        Arguments
        ---------
        interval_definition: dict
            A interval_definition dict

        Notes
        -----
        Unused
        """
        raise NotImplementedError()

    @abstractmethod
    def update_interval_definition(self, interval_definition):
        """Update interval_definition to project configuration

        Arguments
        ---------
        interval_definition_name: str
            Name of the (original) entry
        interval_definition: dict
            The updated interval_definition dict
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_set_scenario_definitions(self, scenario_set_name):
        """Read all scenarios from a certain scenario_set

        Arguments
        ---------
        scenario_set_name: str
            Name of the scenario_set

        Returns
        -------
        list
            A list of scenarios within the specified `scenario_set_name`

        Notes
        -----
        Unused
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_definition(self, scenario_name):
        """Read scenario definition data

        Arguments
        ---------
        scenario_name: str
            Name of the scenario

        Returns
        -------
        dict
            The scenario definition
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_sets(self):
        """Read scenario sets from project configuration

        Returns
        -------
        list
            A list of scenario set dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_set(self, scenario_set_name):
        """Read a scenario_set

        Arguments
        ---------
        scenario_set_name: str
            Name of the scenario_set

        Returns
        -------
        dict
            Scenario set definition
        """
        raise NotImplementedError()

    @abstractmethod
    def write_scenario_set(self, scenario_set):
        """Write scenario_set to project configuration

        Arguments
        ---------
        scenario_set: dict
            A scenario_set dict
        """
        raise NotImplementedError()

    @abstractmethod
    def update_scenario_set(self, scenario_set_name, scenario_set):
        """Update scenario_set to project configuration

        Arguments
        ---------
        scenario_set_name: str
            Name of the (original) entry
        scenario_set: dict
            The updated scenario_set dict
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_scenario_set(self, scenario_set_name):
        """Delete scenario_set from project configuration
        and all scenarios within scenario_set

        Arguments
        ---------
        scenario_set_name: str
            A scenario_set name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenarios(self):
        """Read scenarios from project configuration

        Returns
        -------
        list
            A list of scenario dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario(self, scenario_name):
        """Read a scenario

        Arguments
        ---------
        scenario_name: str
            Name of the scenario

        Returns
        -------
        dict
            A scenario dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def write_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def update_scenario(self, scenario):
        raise NotImplementedError()

    @abstractmethod
    def delete_scenario(self, scenario_name):
        """Delete scenario from project configuration

        Arguments
        ---------
        scenario_name: str
            A scenario name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_data(self, scenario_name, facet_name,
                           spatial_resolution, temporal_resolution, timestep):
        """Read scenario data file

        Arguments
        ---------
        scenario_name: str
            Name of the scenario
        facet_name: str
            Name of the scenario facet to read
        spatial_resolution : str
        temporal_resolution : str
        timestep: int

        Returns
        -------
        data: numpy.ndarray

        Notes
        -----
        Called from smif.data_layer.data_handle
        """
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_sets(self):
        """Read narrative sets from project configuration

        Returns
        -------
        list
            A list of narrative set dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_set(self, narrative_set_name):
        """Read all narratives from a certain narrative_set

        Arguments
        ---------
        narrative_set_name: str
            Name of the narrative_set

        Returns
        -------
        list
            A narrative_set dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_set(self, narrative_set):
        """Write narrative_set to project configuration

        Arguments
        ---------
        narrative_set: dict
            A narrative_set dict
        """
        raise NotImplementedError()

    @abstractmethod
    def update_narrative_set(self, narrative_set):
        """Update narrative_set to project configuration

        Arguments
        ---------
        narrative_set_name: str
            Name of the (original) entry
        narrative_set: dict
            The updated narrative_set dict
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_narrative_set(self, narrative_set_name):
        """Delete narrative_set from project configuration

        Arguments
        ---------
        narrative_set_name: str
            A narrative_set name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_narratives(self):
        """Read narrative sets from project configuration

        Returns
        -------
        list
            A list of narrative set dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_narrative(self, narrative_name):
        """Read all narratives from a certain narrative

        Arguments
        ---------
        narrative_name: str
            Name of the narrative

        Returns
        -------
        list
            A narrative dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def write_narrative(self, narrative):
        """Write narrative to project configuration

        Arguments
        ---------
        narrative: dict
            A narrative dict
        """
        raise NotImplementedError()

    @abstractmethod
    def update_narrative(self, narrative):
        """Update narrative to project configuration

        Arguments
        ---------
        narrative_name: str
            Name of the (original) entry
        narrative: dict
            The updated narrative dict
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_narrative(self, narrative_name):
        """Delete narrative from project configuration

        Arguments
        ---------
        narrative_name: str
            A narrative name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_data(self, narrative_name):
        """Read narrative data file

        Arguments
        ---------
        narrative_name: str
            Name of the narrative

        Returns
        -------
        list
            A list with dictionaries containing the contents of 'narrative_name' data file
        """
        raise NotImplementedError()

    @abstractmethod
    def read_coefficients(self, source_name, destination_name):
        """Reads coefficients from file on disk

        Coefficients are uniquely identified by their source/destination names

        Arguments
        ---------
        source_name : str
            The name of a ResolutionSet
        destination_name : str
            The name of a ResolutionSet

        Notes
        -----
        Both `source_name` and `destination_name` should reference names of
        elements from the same ResolutionSet (e.g. both spatial or temporal
        resolutions)

        Called from smif.convert.register

        """
        raise NotImplementedError

    @abstractmethod
    def write_coefficients(self, source_name, destination_name, data):
        """Writes coefficients to file on disk

        Coefficients are uniquely identified by their source/destination names

        Arguments
        ---------
        source_name : str
            The name of a ResolutionSet
        destination_name : str
            The name of a ResolutionSet
        data : numpy.ndarray

        Notes
        -----
        Both `source_name` and `destination_name` should reference names of
        elements from the same ResolutionSet (e.g. both spatial or temporal
        resolutions)

        Called from smif.convert.register
        """
        raise NotImplementedError()

    @abstractmethod
    def read_results(self, modelrun_name, model_name, output_name, spatial_resolution,
                     temporal_resolution, timestep=None, modelset_iteration=None,
                     decision_iteration=None):
        """Return results of a `model_name` in `modelrun_name` for a given `output_name`

        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_name : str
        spatial_resolution : str
        temporal_resolution : str
        timestep : int, default=None
        modelset_iteration : int, default=None
        decision_iteration : int, default=None

        Returns
        -------
        data: numpy.ndarray

        Notes
        -----
        Called from smif.data_layer.data_handle
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_warm_start(self, modelrun_id):
        """Copy the results from the previous modelrun if available

        The method allows a previous unsuccessful modelrun to 'warm start' a new
        model run from a later timestep. Model results are recovered from the
        timestep that the previous modelrun was run until, and the new model
        run runs from the returned timestep

        Parameters
        ----------
        modelrun_id: str
            The name of the modelrun to recover

        Returns
        -------
        int
            The timestep to which the data store was recovered

        Notes
        -----
        Called from smif.controller.execute
        """
        raise NotImplementedError()

    @abstractmethod
    def write_results(self, modelrun_name, model_name, output_name, data, spatial_resolution,
                      temporal_resolution, timestep=None, modelset_iteration=None,
                      decision_iteration=None):
        """Write results of a `model_name` in `modelrun_name` for a given `output_name`

        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_name : str
        data : numpy.ndarray
        spatial_resolution : str
        temporal_resolution : str
        timestep : int, optional
        modelset_iteration : int, optional
        decision_iteration : int, optional

        Notes
        -----
        Called from smif.data_layer.data_handle
        """
        raise NotImplementedError()

    def read_parameters(self, modelrun_name, model_name):
        """Read global and model-specific parameter values for a given modelrun
        and model.

        Notes
        -----
        Called from smif.data_layer.data_handle
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
    def read_strategies(self):
        raise NotImplementedError

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
