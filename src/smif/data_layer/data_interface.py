"""This module provides a common data interface to smif



Raises
------
SmifDataNotFoundError
    If data cannot be found in the store when try to read from the store
SmifDataExistsError
    If data already exists in the store when trying to write to the store
    (use an update method instead)
SmifDataMismatchError
    Data presented to read, write and update methods is in the
    incorrect format or of wrong dimensions to that expected
SmifDataReadError
    When unable to read data e.g. unable to handle file type or connect
    to database
"""
import copy
from abc import ABCMeta, abstractmethod
from functools import reduce
from logging import getLogger

import numpy as np
from smif.data_layer.data_array import DataArray
from smif.exception import SmifDataMismatchError, SmifDataNotFoundError
from smif.metadata import Spec


class DataInterface(metaclass=ABCMeta):
    """Abstract base class to define common data interface
    """
    def __init__(self):
        self.logger = getLogger(__name__)

    # region Model runs
    @abstractmethod
    def read_model_runs(self):
        """Read all system-of-system model runs

        Returns
        -------
        list
            A list of model_run dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_model_run(self, model_run_name):
        """Read a system-of-system model run

        Arguments
        ---------
        model_run_name: str
            A model_run name

        Returns
        -------
        model_run: dict
            A model_run dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def write_model_run(self, model_run):
        """Write system-of-system model run

        Arguments
        ---------
        model_run: dict
            A model_run dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def update_model_run(self, model_run_name, model_run):
        """Update system-of-system model run

        Arguments
        ---------
        model_run_name: str
            A model_run name
        model_run: dict
            A model_run dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_model_run(self, model_run_name):
        """Delete a system-of-system model run

        Arguments
        ---------
        model_run_name: str
            A model_run name
        """
        raise NotImplementedError()
    # endregion

    # region System-of-systems models
    @abstractmethod
    def read_sos_models(self):
        """Read all system-of-system models

        Returns
        -------
        list
            A list of sos_model dicts
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
        raise NotImplementedError()

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
    # endregion

    # region Sector models
    @abstractmethod
    def read_sector_models(self, skip_coords=False):
        """Read all sector models

        sector_models.yml

        Returns
        -------
        list
            A list of sector_model dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_sector_model(self, sector_model_name, skip_coords=False):
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
    # endregion

    # region Strategies
    @abstractmethod
    def read_strategies(self, modelrun_name):
        """Read strategies for a given model_run

        Arguments
        ---------
        modelrun_name : str
            Name of the model run for which to read the strategies

        Returns
        -------
        list
            List of strategy definition dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def write_strategies(self, modelrun_name, strategies):
        """Write strategies for a given model_run

        Arguments
        ---------
        modelrun_name : str
            Name of the model run for which to read the strategies
        strategies : list[dict]
            List of strategy definitions
        """
    # endregion

    # region Interventions
    @abstractmethod
    def read_interventions(self, sector_model_name):
        """Read interventions data for `sector_model_name`

        Returns
        -------
        dict of dict
            A dict of intervention dictionaries containing intervention
            attributes keyed by intervention name
        """
        raise NotImplementedError()
    # endregion

    # region Initial Conditions
    @abstractmethod
    def read_initial_conditions(self, sector_model_name):
        """Read historical interventions for `sector_model_name`

        Returns
        -------
        list
            A list of historical interventions
        """
        raise NotImplementedError()

    def read_all_initial_conditions(self, model_run_name):
        """A list of all historical interventions

        Returns
        -------
        list
        """
        historical_interventions = []
        sos_model_name = self.read_model_run(model_run_name)['sos_model']
        sector_models = self.read_sos_model(sos_model_name)['sector_models']
        for sector_model_name in sector_models:
            historical_interventions.extend(
                self.read_initial_conditions(sector_model_name)
            )
        return historical_interventions
    # endregion

    # region State
    @abstractmethod
    def read_state(self, modelrun_name, timestep, decision_iteration=None):
        """Read list of (name, build_year) for a given modelrun, timestep,
        decision

        Arguments
        ---------
        modelrun_name : str
        timestep: int
        decision_iteration : int, default=None
        """
        raise NotImplementedError()

    @abstractmethod
    def write_state(self, state, modelrun_name, timestep, decision_iteration=None):
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
    # endregion

    # region Units
    @abstractmethod
    def read_unit_definitions(self):
        """Reads custom unit definitions

        Returns
        -------
        list
            List of str which are valid Pint unit definitions
        """
        raise NotImplementedError()
    # endregion

    # region Dimensions
    @abstractmethod
    def read_dimensions(self):
        """Read dimensions from project configuration

        Returns
        -------
        list
            A list of dimension dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_dimension(self, dimension_name):
        """Return dimension

        Arguments
        ---------
        dimension_name: str
            Name of the dimension

        Returns
        -------
        dict
            A dimension definition (including elements)
        """
        raise NotImplementedError()

    @abstractmethod
    def write_dimension(self, dimension):
        """Write dimension to project configuration

        Arguments
        ---------
        dimension: dict
            A dimension dict

        Notes
        -----
        Unused
        """
        raise NotImplementedError()

    @abstractmethod
    def update_dimension(self, dimension_name, dimension):
        """Update dimension

        Arguments
        ---------
        dimension_name: str
            Name of the (original) entry
        dimension: dict
            The updated dimension dict

        Notes
        -----
        Unused
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_dimension(self, dimension_name):
        """Delete dimension

        Arguments
        ---------
        dimension_name: str
            Name of the (original) entry

        Notes
        -----
        Unused
        """
        raise NotImplementedError()
    # endregion

    # region Conversion coefficients
    @abstractmethod
    def read_coefficients(self, source_spec, destination_spec):
        """Reads coefficients from the store

        Coefficients are uniquely identified by their source/destination specs.
        This method and `write_coefficients` implement caching of conversion
        coefficients between dimensions.

        Arguments
        ---------
        source_spec : smif.metadata.Spec
        destination_spec : smif.metadata.Spec

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.

        """
        raise NotImplementedError

    @abstractmethod
    def write_coefficients(self, source_spec, destination_spec, data):
        """Writes coefficients to the store

        Coefficients are uniquely identified by their source/destination specs.
        This method and `read_coefficients` implement caching of conversion
        coefficients between dimensions.

        Arguments
        ---------
        source_spec : smif.metadata.Spec
        destination_spec : smif.metadata.Spec
        data : numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """
        raise NotImplementedError()
    # endregion

    # region Scenarios
    @abstractmethod
    def read_scenarios(self, skip_coords=False):
        """Read scenarios from project configuration

        Returns
        -------
        list
            A list of scenario dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario(self, scenario_name, skip_coords=False):
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
        """Write scenario to project configuration

        Arguments
        ---------
        scenario: dict
            A scenario dict
        """
        raise NotImplementedError()

    @abstractmethod
    def update_scenario(self, scenario_name, scenario):
        """Update scenario to project configuration

        Arguments
        ---------
        scenario_name: str
            Name of the (original) entry
        scenario: dict
            The updated scenario dict
        """
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
    def read_scenario_variants(self, scenario_name):
        """Read scenarios from project configuration

        Returns
        -------
        list
            A list of scenario dicts
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_variant(self, scenario_name, variant_name):
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
    def write_scenario_variant(self, scenario_name, variant):
        """Write scenario to project configuration

        Arguments
        ---------
        scenario: dict
            A scenario dict
        """
        raise NotImplementedError()

    @abstractmethod
    def update_scenario_variant(self, scenario_name, variant_name, variant):
        """Update scenario to project configuration

        Arguments
        ---------
        scenario_name: str
            Name of the (original) entry
        scenario: dict
            The updated scenario dict
        """
        raise NotImplementedError()

    @abstractmethod
    def delete_scenario_variant(self, scenario_name, variant_name):
        """Delete scenario from project configuration

        Arguments
        ---------
        scenario_name: str
            A scenario name
        """
        raise NotImplementedError()

    @abstractmethod
    def read_scenario_variant_data(self, scenario_name, variant_name, variable, timestep=None):
        """Read scenario data file

        Arguments
        ---------
        scenario_name: str
            Name of the scenario
        variant_name: str
            Name of the scenario variant
        variable: str
            Name of the variable (facet)
        timestep: int (optional)
            If None, read data for all timesteps

        Returns
        -------
        data: numpy.ndarray

        Notes
        -----
        Called from smif.data_layer.data_handle
        """
        raise NotImplementedError()

    @abstractmethod
    def write_scenario_variant_data(self, scenario_name, variant_name, data_array,
                                    timestep=None):
        """Write scenario data file

        Arguments
        ---------
        scenario_name : str
            Name of the scenario
        variant_name : str
            Name of the scenario variant
        data_array : smif.data_layer.data_array.DataArray
            Contains the annotated data
        timestep: int (optional)
            If None, read data for all timesteps

        Notes
        -----
        Called from smif.data_layer.data_handle
        """
        raise NotImplementedError()
    # endregion

    # region Narratives
    @abstractmethod
    def _read_narrative(self, sos_model_name, narrative_name):
        """Read narrative from sos_model

        Arguments
        ---------
        sos_model_name : str
            The name of the sos_model to which the narrative belongs
        narrative_name: str
            Name of the narrative
        """
        raise NotImplementedError()

    @abstractmethod
    def read_narrative_variant_data(self, sos_model_name, narrative_name,
                                    variant_name, variable, timestep=None):
        """Read narrative data file

        Arguments
        ---------
        sos_model_name : str
            The name of the sos_model to which the narrative belongs
        narrative_name: str
            Name of the narrative
        variant_name: str
            Narrative variant to use
        variable: str
            Variable (parameter) to read
        timestep: int (optional)
            Timestep

        Returns
        -------
        smif.data_layer.data_array.DataArray
        """
        raise NotImplementedError()

    @abstractmethod
    def write_narrative_variant_data(self, data, sos_model_name, narrative_name,
                                     data_array, timestep=None):
        """Read narrative data file

        Arguments
        ---------
        data : numpy.ndarray
            The data to write which must be of matching directions to the Spec
            of ``variable``
        sos_model_name : str
            The name of the sos_model to which the narrative belongs
        narrative_name: str
            Name of the narrative
        data_array : smif.data_layer.data_array.DataArray
            Contains the annotated data to write
        timestep: int (optional)
            Timestep

        """
        raise NotImplementedError()
    # endregion

    # region Results
    @abstractmethod
    def read_results(self, modelrun_name, model_name, output_spec, timestep=None,
                     decision_iteration=None):
        """Return results of a `model_name` in `modelrun_name` for a given `output_name`

        Parameters
        ----------
        modelrun_id : str
        model_name : str
        output_spec: smif.metadata.Spec
        timestep : int, default=None
        decision_iteration : int, default=None

        Returns
        -------
        smif.data_layer.data_array.DataArray

        Notes
        -----
        Called from smif.data_layer.data_handle
        """
        raise NotImplementedError()

    @abstractmethod
    def write_results(self, data_array, modelrun_name, model_name, timestep=None,
                      decision_iteration=None):
        """Write results of a `model_name` in `modelrun_name` for a given `output_name`

        Parameters
        ----------
        data_array : smif.data_layer.data_array.DataArray
        modelrun_id : str
        model_name : str
        timestep : int, optional
        decision_iteration : int, optional

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
    # endregion

    # region Common methods
    @staticmethod
    def ndarray_to_data_list(data_array, timestep=None):
        """Convert :class:`numpy.ndarray` to list of observations

        Parameters
        ----------
        data_array : smif.data_layer.data_array.DataArray
        timestep : int, default=None

        Returns
        -------
        observations : list of dict
            Each dict has keys: one for the variable name, one for each dimension in spec.dims,
            and optionally one for the given timestep
        """
        observations = []

        data = data_array.as_ndarray()
        spec = data_array.spec

        for indices, value in np.ndenumerate(data):
            obs = {}
            obs[spec.name] = value
            for dim, idx in zip(spec.dims, indices):
                obs[dim] = spec.dim_coords(dim).elements[idx]['name']
                if timestep:
                    obs['timestep'] = timestep
            observations.append(obs)
        return observations

    @staticmethod
    def data_list_to_ndarray(observations, spec):
        """Convert list of observations to a ``DataArray``

        Parameters
        ----------
        observations : list of dict
            Required keys for each dict are:
            - one key to match spec.name
            - one key per dimension in spec.dims
        spec : smif.metadata.Spec

        Returns
        -------
        smif.data_layer.data_array.DataArray

        Raises
        ------
        KeyError
            If an observation is missing a required key
        ValueError
            If an observation region or interval is not in region_names or
            interval_names
        SmifDataNotFoundError
            If the observations don't include data for any dimension
            combination
        SmifDataMismatchError
            If the dimension coordinate ids do not
            match the observations
        """
        DataInterface._validate_observations(observations, spec)

        data = np.full(spec.shape, np.nan, dtype=spec.dtype)

        for obs in observations:
            indices = []
            for dim in spec.dims:
                key = obs[dim]  # name (id/label) of coordinate element along dimension
                idx = spec.dim_coords(dim).ids.index(key)  # index of name in dim elements
                indices.append(idx)
            data[tuple(indices)] = obs[spec.name]

        return DataArray(spec, data)

    @staticmethod
    def _validate_observations(observations, spec):
        if len(observations) != reduce(lambda x, y: x * y, spec.shape, 1):
            msg = "Number of observations ({}) is not equal to product of {}"
            raise SmifDataMismatchError(
                msg.format(len(observations), spec.shape)
            )
        DataInterface._validate_observation_keys(observations, spec)
        for dim in spec.dims:
            DataInterface._validate_observation_meta(
                observations,
                spec.dim_coords(dim).ids,
                dim
            )

    @staticmethod
    def _validate_observation_keys(observations, spec):
        for obs in observations:
            if spec.name not in obs:
                raise KeyError(
                    "Observation missing variable key ({}): {}".format(spec.name, obs))
            for dim in spec.dims:
                if dim not in obs:
                    raise KeyError(
                        "Observation missing dimension key ({}): {}".format(dim, obs))

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
            raise SmifDataNotFoundError(
                "Missing values for {}s: {}".format(meta_name, list(missing)))

    @staticmethod
    def _skip_coords(config, keys):
        """Given a config dict and list of top-level keys for lists of specs,
        delete coords from each spec in each list.
        """
        config = copy.deepcopy(config)
        for key in keys:
            for spec in config[key]:
                try:
                    del spec['coords']
                except KeyError:
                    pass
        return config

    def _read_narrative_variable_spec(self, sos_model_name, narrative_name, variable):
        # Read spec from narrative->provides->variable
        narrative = self._read_narrative(sos_model_name, narrative_name)
        model_name = self._key_from_list(variable, narrative['provides'])
        if not model_name:
            msg = "Cannot identify source of Spec for variable '{}'"
            raise SmifDataNotFoundError(msg.format(variable))
        parameters = self.read_sector_model(model_name)['parameters']
        return self._get_spec_from_provider(parameters, variable)

    def _get_spec_from_provider(self, config_list, variable_name):
        """Gets a Spec definition from a scenario definition

        Arguments
        ---------
        config_list : list of dict
            A list of spec dicts
        variable_name : str
            The name of the variable for which to find the spec

        Returns
        -------
        smif.metadata.Spec
        """
        spec = self._pick_from_list(config_list, variable_name)
        if spec is not None:
            self._set_item_coords(spec)
            return Spec.from_dict(spec)
        else:
            msg = "Could not find spec definition for '{}'"
            raise SmifDataNotFoundError(msg.format(variable_name))

    def _set_item_coords(self, item):
        """If dims exists and is not empty
        """
        if 'dims' in item and item['dims']:
            item['coords'] = {
                dim: self.read_dimension(dim)['elements']
                for dim in item['dims']
            }

    @staticmethod
    def _pick_from_list(list_of_dicts, name):
        for item in list_of_dicts:
            if 'name' in item and item['name'] == name:
                return item
        return None

    @staticmethod
    def _key_from_list(name, dict_of_lists):
        for key, items in dict_of_lists.items():
            if name in items:
                return key
        return None
    # endregion
