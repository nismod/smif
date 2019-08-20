"""The store provides a common data interface to smif configuration, data and metadata.

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
import itertools
import logging
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from operator import itemgetter
from os.path import splitext
from typing import Dict, List, Optional, Union

import numpy as np  # type: ignore
from smif.data_layer import DataArray
from smif.data_layer.abstract_data_store import DataStore
from smif.data_layer.abstract_metadata_store import MetadataStore
from smif.data_layer.file import (CSVDataStore, FileMetadataStore,
                                  ParquetDataStore, YamlConfigStore)
from smif.data_layer.validate import (validate_sos_model_config,
                                      validate_sos_model_format)
from smif.exception import SmifDataError, SmifDataNotFoundError
from smif.metadata.spec import Spec


class Store():
    """Common interface to data store, composed of config, metadata and data store
    implementations.

    Parameters
    ----------
    config_store: ~smif.data_layer.abstract_config_store.ConfigStore
    metadata_store: ~smif.data_layer.abstract_metadata_store.MetadataStore
    data_store: ~smif.data_layer.abstract_data_store.DataStore
    """

    def __init__(self, config_store, metadata_store: MetadataStore,
                 data_store: DataStore, model_base_folder="."):
        self.logger = logging.getLogger(__name__)
        self.config_store = config_store
        self.metadata_store = metadata_store
        self.data_store = data_store
        # base folder for any relative paths to models
        self.model_base_folder = str(model_base_folder)

    @classmethod
    def from_dict(cls, config):
        """Create Store from configuration dict
        """

        try:
            interface = config['interface']
        except KeyError:
            logging.warning('No interface provided for Results().  Assuming local_csv')
            interface = 'local_csv'

        try:
            directory = config['dir']
        except KeyError:
            logging.warning("No directory provided for Results().  Assuming '.'")
            directory = '.'

        # Check that the directory is valid
        if not os.path.isdir(directory):
            raise ValueError('Expected {} to be a valid directory'.format(directory))

        if interface == 'local_csv':
            data_store = CSVDataStore(directory)
        elif interface == 'local_parquet':
            data_store = ParquetDataStore(directory)
        else:
            raise ValueError(
                'Unsupported interface "{}". Supply local_csv or local_parquet'.format(
                    interface))

        return cls(
            config_store=YamlConfigStore(directory),
            metadata_store=FileMetadataStore(directory),
            data_store=data_store,
            model_base_folder=directory
        )

    #
    # CONFIG
    #

    # region Model runs
    def read_model_runs(self):
        """Read all system-of-system model runs

        Returns
        -------
        list[~smif.controller.modelrun.ModelRun]
        """
        return sorted(self.config_store.read_model_runs(), key=itemgetter('name'))

    def read_model_run(self, model_run_name):
        """Read a system-of-system model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        ~smif.controller.modelrun.ModelRun
        """
        return self.config_store.read_model_run(model_run_name)

    def write_model_run(self, model_run):
        """Write system-of-system model run

        Parameters
        ----------
        model_run : ~smif.controller.modelrun.ModelRun
        """
        self.config_store.write_model_run(model_run)

    def update_model_run(self, model_run_name, model_run):
        """Update system-of-system model run

        Parameters
        ----------
        model_run_name : str
        model_run : ~smif.controller.modelrun.ModelRun
        """
        self.config_store.update_model_run(model_run_name, model_run)

    def delete_model_run(self, model_run_name):
        """Delete a system-of-system model run

        Parameters
        ----------
        model_run_name : str
        """
        self.config_store.delete_model_run(model_run_name)

    # endregion

    # region System-of-systems models
    def read_sos_models(self):
        """Read all system-of-system models

        Returns
        -------
        list[~smif.model.sos_model.SosModel]
        """
        return sorted(self.config_store.read_sos_models(), key=itemgetter('name'))

    def read_sos_model(self, sos_model_name):
        """Read a specific system-of-system model

        Parameters
        ----------
        sos_model_name : str

        Returns
        -------
        ~smif.model.sos_model.SosModel
        """
        return self.config_store.read_sos_model(sos_model_name)

    def write_sos_model(self, sos_model):
        """Write system-of-system model

        Parameters
        ----------
        sos_model : ~smif.model.sos_model.SosModel
        """
        validate_sos_model_format(sos_model)
        self.config_store.write_sos_model(sos_model)

    def update_sos_model(self, sos_model_name, sos_model):
        """Update system-of-system model

        Parameters
        ----------
        sos_model_name : str
        sos_model : ~smif.model.sos_model.SosModel
        """
        models = self.config_store.read_models()
        scenarios = self.config_store.read_scenarios()
        validate_sos_model_config(sos_model, models, scenarios)

        self.config_store.update_sos_model(sos_model_name, sos_model)

    def delete_sos_model(self, sos_model_name):
        """Delete a system-of-system model

        Parameters
        ----------
        sos_model_name : str
        """
        self.config_store.delete_sos_model(sos_model_name)

    # endregion

    # region Models
    def read_models(self, skip_coords=False):
        """Read all models

        Returns
        -------
        list[~smif.model.model.Model]
        """
        models = sorted(self.config_store.read_models(), key=itemgetter('name'))
        if not skip_coords:
            models = [
                self._add_coords(model, ('inputs', 'outputs', 'parameters'))
                for model in models
            ]
        return models

    def read_model(self, model_name, skip_coords=False):
        """Read a model

        Parameters
        ----------
        model_name : str

        Returns
        -------
        ~smif.model.model.Model
        """
        model = self.config_store.read_model(model_name)
        if not skip_coords:
            model = self._add_coords(model, ('inputs', 'outputs', 'parameters'))
        return model

    def write_model(self, model):
        """Write a model

        Parameters
        ----------
        model : ~smif.model.model.Model
        """
        self.config_store.write_model(model)

    def update_model(self, model_name, model):
        """Update a model

        Parameters
        ----------
        model_name : str
        model : ~smif.model.model.Model
        """
        self.config_store.update_model(model_name, model)

    def delete_model(self, model_name):
        """Delete a model

        Parameters
        ----------
        model_name : str
        """
        self.config_store.delete_model(model_name)

    # endregion

    # region Scenarios
    def read_scenarios(self, skip_coords=False):
        """Read scenarios

        Returns
        -------
        list[~smif.model.ScenarioModel]
        """
        scenarios = sorted(self.config_store.read_scenarios(), key=itemgetter('name'))
        if not skip_coords:
            scenarios = [
                self._add_coords(scenario, ['provides'])
                for scenario in scenarios
            ]
        return scenarios

    def read_scenario(self, scenario_name, skip_coords=False):
        """Read a scenario

        Parameters
        ----------
        scenario_name : str

        Returns
        -------
        ~smif.model.ScenarioModel
        """
        scenario = self.config_store.read_scenario(scenario_name)
        if not skip_coords:
            scenario = self._add_coords(scenario, ['provides'])
        return scenario

    def write_scenario(self, scenario):
        """Write scenario

        Parameters
        ----------
        scenario : ~smif.model.ScenarioModel
        """
        self.config_store.write_scenario(scenario)

    def update_scenario(self, scenario_name, scenario):
        """Update scenario

        Parameters
        ----------
        scenario_name : str
        scenario : ~smif.model.ScenarioModel
        """
        self.config_store.update_scenario(scenario_name, scenario)

    def delete_scenario(self, scenario_name):
        """Delete scenario from project configuration

        Parameters
        ----------
        scenario_name : str
        """
        self.config_store.delete_scenario(scenario_name)

    def prepare_scenario(self, scenario_name, list_of_variants):
        """ Modify {scenario_name} config file to include multiple
        scenario variants.

        Parameters
        ----------
        scenario_name: str
        list_of_variants: list[int] - indices of scenario variants
        """
        scenario = self.read_scenario(scenario_name)
        # Check that template scenario file does not define more than one variant
        if not scenario['variants'] or len(scenario['variants']) > 1:
            raise SmifDataError("Template scenario file must define one"
                                " unique template variant.")

        # Read variant defined in template scenario file
        variant_template_name = scenario['variants'][0]['name']
        base_variant = self.read_scenario_variant(scenario_name, variant_template_name)
        self.delete_scenario_variant(scenario_name, variant_template_name)

        # Read template names of scenario variant data files
        output_filenames = {}  # output_name => (base, ext)
        # root is a dict. keyed on scenario outputs.
        # Entries contain the root of the variants filenames
        for output in scenario['provides']:
            output_name = output['name']
            base, ext = splitext(base_variant['data'][output_name])
            output_filenames[output_name] = base, ext
        # Now modify scenario file
        for variant_number in list_of_variants:
            # Copying the variant dict is required when underlying config_store
            # is an instance of MemoryConfigStore, which attribute _scenarios holds
            # a reference to the variant object passed to update or
            # write_scenario_variant
            variant = deepcopy(base_variant)
            variant['name'] = '{}_{:03d}'.format(scenario_name, variant_number)
            variant['description'] = '{} variant number {:03d}'.format(
                scenario_name, variant_number)
            for output_name, (base, ext) in output_filenames.items():
                variant['data'][output_name] = '{}{:03d}{}'.format(base, variant_number, ext)
            self.write_scenario_variant(scenario_name, variant)

    def prepare_model_runs(self, model_run_name, scenario_name,
                           first_var, last_var):
        """Write multiple model run config files corresponding to multiple
        scenario variants of {scenario_name}, based on template {model_run_name}
           Write batchfile containing each of the generated model runs

        Parameters
        ----------
        model_run_name: str
        scenario_name: str
        first_var: int - between 0 and number of variants-1
        last_var: int - between first_var and number of variants-1
        """

        model_run = self.read_model_run(model_run_name)
        scenario = self.read_scenario(scenario_name)

        # read strategies from config store (Store.read_strategies pulls together data on
        # interventions as well, which we don't need here)
        config_strategies = self.config_store.read_strategies(model_run_name)
        # Open batchfile
        f_handle = open(model_run_name + '.batch', 'w')
        # For each variant model_run, write a new model run file with corresponding
        # scenario variant and update batchfile
        for variant in scenario['variants'][first_var:last_var + 1]:
            variant_model_run_name = model_run_name + '_' + variant['name']
            model_run_copy = deepcopy(model_run)
            model_run_copy['name'] = variant_model_run_name
            model_run_copy['scenarios'][scenario_name] = variant['name']

            self.write_model_run(model_run_copy)
            self.config_store.write_strategies(variant_model_run_name, config_strategies)
            f_handle.write(model_run_name + '_' + variant['name'] + '\n')

        # Close batchfile
        f_handle.close()

    # endregion

    # region Scenario Variants
    def read_scenario_variants(self, scenario_name):
        """Read variants of a given scenario

        Parameters
        ----------
        scenario_name : str

        Returns
        -------
        list[dict]
        """
        scenario_variants = self.config_store.read_scenario_variants(scenario_name)
        return sorted(scenario_variants, key=itemgetter('name'))

    def read_scenario_variant(self, scenario_name, variant_name):
        """Read a scenario variant

        Parameters
        ----------
        scenario_name : str
        variant_name : str

        Returns
        -------
        dict
        """
        return self.config_store.read_scenario_variant(scenario_name, variant_name)

    def write_scenario_variant(self, scenario_name, variant):
        """Write scenario to project configuration

        Parameters
        ----------
        scenario_name : str
        variant : dict
        """
        self.config_store.write_scenario_variant(scenario_name, variant)

    def update_scenario_variant(self, scenario_name, variant_name, variant):
        """Update scenario to project configuration

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        variant : dict
        """
        self.config_store.update_scenario_variant(scenario_name, variant_name, variant)

    def delete_scenario_variant(self, scenario_name, variant_name):
        """Delete scenario from project configuration

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        """
        self.config_store.delete_scenario_variant(scenario_name, variant_name)

    # endregion

    # region Narratives
    def read_narrative(self, sos_model_name, narrative_name):
        """Read narrative from sos_model

        Parameters
        ----------
        sos_model_name : str
        narrative_name : str
        """
        return self.config_store.read_narrative(sos_model_name, narrative_name)

    # endregion

    # region Strategies
    def read_strategies(self, model_run_name):
        """Read strategies for a given model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        list[dict]
        """
        strategies = deepcopy(self.config_store.read_strategies(model_run_name))
        for strategy in strategies:
            if strategy['type'] == 'pre-specified-planning':
                strategy['interventions'] = self.data_store.read_strategy_interventions(
                    strategy)
        return strategies

    def write_strategies(self, model_run_name, strategies):
        """Write strategies for a given model_run

        Parameters
        ----------
        model_run_name : str
        strategies : list[dict]
        """
        self.config_store.write_strategies(model_run_name, strategies)

    def convert_strategies_data(self, model_run_name, tgt_store, noclobber=False):
        strategies = self.read_strategies(model_run_name)
        for strategy in strategies:
            if strategy['type'] == 'pre-specified-planning':
                data_exists = tgt_store.read_strategy_interventions(
                    strategy, assert_exists=True)
                if not(noclobber and data_exists):
                    data = self.read_strategy_interventions(strategy)
                    tgt_store.write_strategy_interventions(strategy, data)

    # endregion

    #
    # METADATA
    #

    # region Units
    def read_unit_definitions(self) -> List[str]:
        """Reads custom unit definitions

        Returns
        -------
        list[str]
            Pint-compatible unit definitions
        """
        return self.metadata_store.read_unit_definitions()

    def write_unit_definitions(self, definitions):
        """Reads custom unit definitions

        Parameters
        ----------
        list[str]
            Pint-compatible unit definitions
        """
        self.metadata_store.write_unit_definitions(definitions)

    # endregion

    # region Dimensions
    def read_dimensions(self, skip_coords=False):
        """Read dimensions

        Returns
        -------
        list[~smif.metadata.coords.Coords]
        """
        return self.metadata_store.read_dimensions(skip_coords)

    def read_dimension(self, dimension_name, skip_coords=False):
        """Return dimension

        Parameters
        ----------
        dimension_name : str

        Returns
        -------
        ~smif.metadata.coords.Coords
            A dimension definition (including elements)
        """
        return self.metadata_store.read_dimension(dimension_name, skip_coords)

    def write_dimension(self, dimension):
        """Write dimension to project configuration

        Parameters
        ----------
        dimension : ~smif.metadata.coords.Coords
        """
        self.metadata_store.write_dimension(dimension)

    def update_dimension(self, dimension_name, dimension):
        """Update dimension

        Parameters
        ----------
        dimension_name : str
        dimension : ~smif.metadata.coords.Coords
        """
        self.metadata_store.update_dimension(dimension_name, dimension)

    def delete_dimension(self, dimension_name):
        """Delete dimension

        Parameters
        ----------
        dimension_name : str
        """
        self.metadata_store.delete_dimension(dimension_name)

    def _add_coords(self, item, keys):
        """Add coordinates to spec definitions on an object
        """
        item = deepcopy(item)
        for key in keys:
            spec_list = item[key]
            for spec in spec_list:
                if 'dims' in spec and spec['dims']:
                    spec['coords'] = {
                        dim: self.read_dimension(dim)['elements']
                        for dim in spec['dims']
                    }
        return item

    # endregion

    #
    # DATA
    #

    # region Scenario Variant Data
    def read_scenario_variant_data(
            self, scenario_name: str, variant_name: str, variable: str,
            timestep: Optional[int] = None, timesteps: Optional[List[int]] = None,
            assert_exists: bool = False) -> Union[DataArray, bool]:
        """Read scenario data file

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        variable : str
        timestep : int

        Returns
        -------
        data : ~smif.data_layer.data_array.DataArray
        """
        variant = self.read_scenario_variant(scenario_name, variant_name)
        key = self._key_from_data(variant['data'][variable], scenario_name, variant_name,
                                  variable)
        scenario = self.read_scenario(scenario_name)
        spec_dict = _pick_from_list(scenario['provides'], variable)
        spec = Spec.from_dict(spec_dict)
        if assert_exists:
            return self.data_store.scenario_variant_data_exists(key)
        else:
            return self.data_store.read_scenario_variant_data(key, spec, timestep, timesteps)

    def write_scenario_variant_data(self, scenario_name, variant_name, data):
        """Write scenario data file

        Parameters
        ----------
        scenario_name : str
        variant_name : str
        data : ~smif.data_layer.data_array.DataArray
        """
        variant = self.read_scenario_variant(scenario_name, variant_name)
        key = self._key_from_data(variant['data'][data.spec.name], scenario_name, variant_name,
                                  data.spec.name)
        self.data_store.write_scenario_variant_data(key, data)

    def convert_scenario_data(self, model_run_name, tgt_store, noclobber=False):
        model_run = self.read_model_run(model_run_name)
        # Convert scenario data for model run
        for scenario_name in model_run['scenarios']:
            for variant in self.read_scenario_variants(scenario_name):
                for variable in variant['data']:
                    data_exists = tgt_store.read_scenario_variant_data(
                        scenario_name, variant['name'], variable, assert_exists=True)
                    if not(noclobber and data_exists):
                        data_array = self.read_scenario_variant_data(
                            scenario_name, variant['name'], variable)
                        tgt_store.write_scenario_variant_data(
                            scenario_name, variant['name'], data_array)
    # endregion

    # region Narrative Data
    def read_narrative_variant_data(self, sos_model_name, narrative_name, variant_name,
                                    parameter_name, timestep=None, assert_exists=False):
        """Read narrative data file

        Parameters
        ----------
        sos_model_name : str
        narrative_name : str
        variant_name : str
        parameter_name : str
        timestep : int (optional)
            If None, read data for all timesteps

        Returns
        -------
        ~smif.data_layer.data_array.DataArray
        """
        sos_model = self.read_sos_model(sos_model_name)

        narrative = _pick_from_list(sos_model['narratives'], narrative_name)

        if narrative is None:
            msg = "Narrative name '{}' does not exist in sos_model '{}'"
            raise SmifDataNotFoundError(msg.format(narrative_name, sos_model_name))

        variant = _pick_from_list(narrative['variants'], variant_name)

        if variant is None:
            msg = "Variant name '{}' does not exist in narrative '{}'"
            raise SmifDataNotFoundError(msg.format(variant_name, narrative_name))

        key = self._key_from_data(variant['data'][parameter_name], narrative_name,
                                  variant_name, parameter_name)

        if assert_exists:
            return self.data_store.narrative_variant_data_exists(key)
        else:
            spec_dict = None
            # find sector model which needs this parameter, to get spec definition
            for model_name, params in narrative['provides'].items():
                if parameter_name in params:
                    sector_model = self.read_model(model_name)
                    spec_dict = _pick_from_list(sector_model['parameters'], parameter_name)
                    break
            # find spec
            if spec_dict is None:
                raise SmifDataNotFoundError("Parameter {} not found in any of {}".format(
                    parameter_name, sos_model['sector_models']))
            spec = Spec.from_dict(spec_dict)

            return self.data_store.read_narrative_variant_data(key, spec, timestep)

    def write_narrative_variant_data(self, sos_model_name, narrative_name, variant_name, data):
        """Read narrative data file

        Parameters
        ----------
        sos_model_name : str
        narrative_name : str
        variant_name : str
        data : ~smif.data_layer.data_array.DataArray
        """
        sos_model = self.read_sos_model(sos_model_name)
        narrative = _pick_from_list(sos_model['narratives'], narrative_name)
        variant = _pick_from_list(narrative['variants'], variant_name)
        key = self._key_from_data(
            variant['data'][data.spec.name], narrative_name, variant_name, data.spec.name)
        self.data_store.write_narrative_variant_data(key, data)

    def convert_narrative_data(self, sos_model_name, tgt_store, noclobber=False):
        sos_model = self.read_sos_model(sos_model_name)
        for narrative in sos_model['narratives']:
            for variant in narrative['variants']:
                for param in variant['data']:
                    data_exists = tgt_store.read_narrative_variant_data(sos_model_name,
                                                                        narrative['name'],
                                                                        variant['name'],
                                                                        param,
                                                                        assert_exists=True)
                if not(noclobber and data_exists):
                    data_array = self.read_narrative_variant_data(sos_model_name,
                                                                  narrative['name'],
                                                                  variant['name'],
                                                                  param)
                    tgt_store.write_narrative_variant_data(sos_model_name, narrative['name'],
                                                           variant['name'], data_array)

    def read_model_parameter_default(self, model_name, parameter_name, assert_exists=False):
        """Read default data for a sector model parameter

        Parameters
        ----------
        model_name : str
        parameter_name : str

        Returns
        -------
        ~smif.data_layer.data_array.DataArray
        """
        model = self.read_model(model_name)
        param = _pick_from_list(model['parameters'], parameter_name)
        spec = Spec.from_dict(param)
        try:
            path = param['default']
        except TypeError:
            raise SmifDataNotFoundError("Parameter {} not found in model {}".format(
                parameter_name, model_name))
        except KeyError:
            path = 'default__{}__{}.csv'.format(model_name, parameter_name)
        key = self._key_from_data(path, model_name, parameter_name)
        if assert_exists:
            return self.data_store.model_parameter_default_data_exists(key)
        else:
            return self.data_store.read_model_parameter_default(key, spec)

    def write_model_parameter_default(self, model_name, parameter_name, data):
        """Write default data for a sector model parameter

        Parameters
        ----------
        model_name : str
        parameter_name : str
        data : ~smif.data_layer.data_array.DataArray
        """
        model = self.read_model(model_name, skip_coords=True)
        param = _pick_from_list(model['parameters'], parameter_name)
        try:
            path = param['default']
        except TypeError:
            raise SmifDataNotFoundError("Parameter {} not found in model {}".format(
                parameter_name, model_name))
        except KeyError:
            path = 'default__{}__{}.csv'.format(model_name, parameter_name)
        key = self._key_from_data(path, model_name, parameter_name)
        self.data_store.write_model_parameter_default(key, data)

    def convert_model_parameter_default_data(self, sector_model_name, tgt_store,
                                             noclobber=False):
        sector_model = self.read_model(sector_model_name)
        for parameter in sector_model['parameters']:
            data_exists = tgt_store.read_model_parameter_default(sector_model_name,
                                                                 parameter['name'],
                                                                 assert_exists=True)
            if not(noclobber and data_exists):
                data_array = self.read_model_parameter_default(sector_model_name,
                                                               parameter['name'])
                tgt_store.write_model_parameter_default(sector_model_name, parameter['name'],
                                                        data_array)
    # endregion

    # region Interventions
    def read_interventions(self, model_name):
        """Read interventions data for `model_name`

        Returns
        -------
        dict[str, dict]
            A dict of intervention dictionaries containing intervention
            attributes keyed by intervention name
        """
        model = self.read_model(model_name, skip_coords=True)
        if model['interventions'] != []:
            return self.data_store.read_interventions(model['interventions'])
        else:
            return {}

    def write_interventions(self, model_name, interventions):
        """Write interventions data for a model

        Parameters
        ----------
        dict[str, dict]
            A dict of intervention dictionaries containing intervention
            attributes keyed by intervention name
        """
        model = self.read_model(model_name)
        model['interventions'] = [model_name + '.csv']
        self.update_model(model_name, model)
        self.data_store.write_interventions(model['interventions'][0], interventions)

    def write_interventions_file(self, model_name, string_id, interventions):
        model = self.read_model(model_name)
        if string_id in model['interventions']:
            self.data_store.write_interventions(string_id, interventions)
        else:
            raise SmifDataNotFoundError("Intervention {} not found for"
                                        " sector model {}.".format(string_id, model_name))

    def read_interventions_file(self, model_name, string_id, assert_exists=False):
        model = self.read_model(model_name)
        if string_id in model['interventions']:
            if assert_exists:
                return self.data_store.interventions_data_exists(string_id)
            else:
                return self.data_store.read_interventions([string_id])
        else:
            raise SmifDataNotFoundError("Intervention {} not found for"
                                        " sector model {}.".format(string_id, model_name))

    def convert_interventions_data(self, sector_model_name, tgt_store, noclobber=False):
        sector_model = self.read_model(sector_model_name)
        for intervention in sector_model['interventions']:
            data_exists = tgt_store.read_interventions_file(sector_model_name,
                                                            intervention,
                                                            assert_exists=True)
            if not(noclobber and data_exists):
                interventions = self.read_interventions_file(sector_model_name, intervention)
                tgt_store.write_interventions_file(
                    sector_model_name, intervention, interventions)

    def read_strategy_interventions(self, strategy, assert_exists=False):
        """Read interventions as defined in a model run strategy
        """
        if assert_exists:
            return self.data_store.strategy_data_exists(strategy)
        else:
            return self.data_store.read_strategy_interventions(strategy)

    def write_strategy_interventions(self, strategy, data):
        """
        Parameters
        ----------
        list[dicts]
        """
        self.data_store.write_strategy_interventions(strategy, data)

    def read_initial_conditions(self, model_name) -> List[Dict]:
        """Read historical interventions for `model_name`

        Returns
        -------
        list[dict]
            A list of historical interventions, with keys 'name' and 'build_year'
        """
        model = self.read_model(model_name)
        if model['initial_conditions'] != []:
            return self.data_store.read_initial_conditions(model['initial_conditions'])
        else:
            return []

    def write_initial_conditions(self, model_name, initial_conditions):
        """Write historical interventions for a model

        Parameters
        ----------
        list[dict]
            A list of historical interventions, with keys 'name' and 'build_year'
        """
        model = self.read_model(model_name)
        model['initial_conditions'] = [model_name + '.csv']
        self.update_model(model_name, model)
        self.data_store.write_initial_conditions(model['initial_conditions'][0],
                                                 initial_conditions)

    def write_initial_conditions_file(self, model_name, string_id, initial_conditions):
        model = self.read_model(model_name)
        if string_id in model['initial_conditions']:
            self.data_store.write_initial_conditions(string_id, initial_conditions)
        else:
            raise SmifDataNotFoundError("Initial condition {} not found for"
                                        " sector model {}.".format(string_id, model_name))

    def read_initial_conditions_file(self, model_name, string_id, assert_exists=False):
        model = self.read_model(model_name)
        if string_id in model['initial_conditions']:
            if assert_exists:
                return self.data_store.initial_conditions_data_exists(string_id)
            else:
                return self.data_store.read_initial_conditions([string_id])
        else:
            raise SmifDataNotFoundError("Initial conditions {} not found for"
                                        " sector model {}.".format(string_id, model_name))

    def convert_initial_conditions_data(self, sector_model_name, tgt_store, noclobber=False):
        sector_model = self.read_model(sector_model_name)
        for initial_condition in sector_model['initial_conditions']:
            data_exists = tgt_store.read_initial_conditions_file(sector_model_name,
                                                                 initial_condition,
                                                                 assert_exists=True)
            if not(noclobber and data_exists):
                initial_conditions = self.read_initial_conditions_file(sector_model_name,
                                                                       initial_condition)
                tgt_store.write_initial_conditions_file(sector_model_name, initial_condition,
                                                        initial_conditions)

    def read_all_initial_conditions(self, model_run_name) -> List[Dict]:
        """A list of all historical interventions

        Returns
        -------
        list[dict]
        """
        historical_interventions = []  # type: List
        model_run = self.read_model_run(model_run_name)
        sos_model_name = model_run['sos_model']
        sos_model = self.read_sos_model(sos_model_name)
        sector_model_names = sos_model['sector_models']
        for sector_model_name in sector_model_names:
            historical_interventions.extend(
                self.read_initial_conditions(sector_model_name)
            )
        return historical_interventions

    # endregion

    # region State
    def read_state(self, model_run_name, timestep, decision_iteration=None) -> List[Dict]:
        """Read list of (name, build_year) for a given model_run, timestep,
        decision

        Parameters
        ----------
        model_run_name : str
        timestep : int
        decision_iteration : int, optional

        Returns
        -------
        list[dict]
        """
        return self.data_store.read_state(model_run_name, timestep, decision_iteration)

    def write_state(self, state, model_run_name, timestep, decision_iteration=None):
        """State is a list of decisions with name and build_year.

        State is output from the DecisionManager

        Parameters
        ----------
        state : list[dict]
        model_run_name : str
        timestep : int
        decision_iteration : int, optional
        """
        self.data_store.write_state(state, model_run_name, timestep, decision_iteration)

    # endregion

    # region Conversion coefficients
    def read_coefficients(self, source_dim: str, destination_dim: str) -> np.ndarray:
        """Reads coefficients from the store

        Coefficients are uniquely identified by their source/destination dimensions.
        This method and `write_coefficients` implement caching of conversion
        coefficients between dimensions.

        Parameters
        ----------
        source_dim : str
            Dimension name
        destination_dim : str
            Dimension name

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """
        return self.data_store.read_coefficients(source_dim, destination_dim)

    def write_coefficients(self, source_dim: str, destination_dim: str, data: np.ndarray):
        """Writes coefficients to the store

        Coefficients are uniquely identified by their source/destination dimensions.
        This method and `read_coefficients` implement caching of conversion
        coefficients between dimensions.

        Parameters
        ----------
        source_dim : str
            Dimension name
        destination_dim : str
            Dimension name
        data : numpy.ndarray

        Notes
        -----
        To be called from :class:`~smif.convert.adaptor.Adaptor` implementations.
        """
        self.data_store.write_coefficients(source_dim, destination_dim, data)

    # endregion

    # region Results
    def read_results(self,
                     model_run_name: str,
                     model_name: str,
                     output_spec: Spec,
                     timestep: Optional[int] = None,
                     decision_iteration: Optional[int] = None) -> DataArray:
        """Return results of a `model_name` in `model_run_name` for a given `output_name`

        Parameters
        ----------
        model_run_name : str
        model_name : str
        output_spec : smif.metadata.Spec
        timestep : int, default=None
        decision_iteration : int, default=None

        Returns
        -------
        ~smif.data_layer.data_array.DataArray
        """
        return self.data_store.read_results(
            model_run_name, model_name, output_spec, timestep, decision_iteration)

    def write_results(self, data_array, model_run_name, model_name, timestep=None,
                      decision_iteration=None):
        """Write results of a `model_name` in `model_run_name` for a given `output_name`

        Parameters
        ----------
        data_array : ~smif.data_layer.data_array.DataArray
        model_run_id : str
        model_name : str
        timestep : int, optional
        decision_iteration : int, optional
        """
        self.data_store.write_results(
            data_array, model_run_name, model_name, timestep, decision_iteration)

    def delete_results(self, model_run_name, model_name, output_name, timestep=None,
                       decision_iteration=None):
        """Delete results for a single timestep/iteration of a model output in a model run

        Parameters
        ----------
        model_run_name : str
        model_name : str
        output_name : str
        timestep : int, default=None
        decision_iteration : int, default=None
        """
        self.data_store.delete_results(
            model_run_name, model_name, output_name, timestep, decision_iteration)

    def clear_results(self, model_run_name):
        """Clear all results from a single model run

        Parameters
        ----------
        model_run_name : str
        """
        available = self.available_results(model_run_name)
        for timestep, decision_iteration, model_name, output_name in available:
            self.data_store.delete_results(
                model_run_name, model_name, output_name, timestep, decision_iteration)

    def available_results(self, model_run_name):
        """List available results from a model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        list[tuple]
             Each tuple is (timestep, decision_iteration, model_name, output_name)
        """
        return self.data_store.available_results(model_run_name)

    def completed_jobs(self, model_run_name):
        """List completed jobs from a model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        list[tuple]
             Each tuple is (timestep, decision_iteration, model_name)
        """
        available_results = self.available_results(model_run_name)  # {(t, d, model, output)}
        model_outputs = self.expected_model_outputs(model_run_name)  # [(model, output)]
        completed_jobs = self.filter_complete_available_results(
            available_results, model_outputs)
        return completed_jobs

    @staticmethod
    def filter_complete_available_results(available_results, expected_model_outputs):
        """Filter available results from a model run to include only complete timestep/decision
        iteration combinations

        Parameters
        ----------
        available_results: list[tuple]
            List of (timestep, decision_iteration, model_name, output_name)
        expected_model_outputs: list[tuple]
            List or set of (model_name, output_name)

        Returns
        -------
        list[tuple]
             Each tuple is (timestep, decision_iteration, model_name)
        """
        expected_model_outputs = set(expected_model_outputs)
        model_names = {model_name for model_name, _ in expected_model_outputs}
        model_outputs_by_td = defaultdict(set)
        for timestep, decision, model_name, output_name in available_results:
            model_outputs_by_td[(timestep, decision)].add((model_name, output_name))

        completed_jobs = []
        for (timestep, decision), td_model_outputs in model_outputs_by_td.items():
            if td_model_outputs == expected_model_outputs:
                for model_name in model_names:
                    completed_jobs.append((timestep, decision, model_name))
        return completed_jobs

    def expected_model_outputs(self, model_run_name):
        """List expected model outputs from a model run

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        list[tuple]
             Each tuple is (model_name, output_name)
        """
        model_run = self.read_model_run(model_run_name)
        sos_model_name = model_run['sos_model']
        sos_config = self.read_sos_model(sos_model_name)

        # For each model, get the outputs and create (model_name, output_name) tuples
        expected_model_outputs = []
        for model_name in sos_config['sector_models']:
            model_config = self.read_model(model_name)
            for output in model_config['outputs']:
                expected_model_outputs.append((model_name, output['name']))

        return expected_model_outputs

    def prepare_warm_start(self, model_run_name):
        """Copy the results from the previous model_run if available

        The method allows a previous unsuccessful model_run to 'warm start' a new model run
        from a later timestep. Model results are recovered from the timestep that the previous
        model_run was run until, and the new model run runs from the returned timestep

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        int The timestep to which the data store was recovered

        Notes
        -----
        Called from smif.controller.execute
        """
        available_results = self.available_results(model_run_name)
        if available_results:
            max_timestep = max(
                timestep for
                timestep, decision_iteration, model_name, output_name in available_results
            )
            # could explicitly clear results for max timestep
        else:
            max_timestep = None
        return max_timestep

    def canonical_available_results(self, model_run_name):
        """List the results that are available from a model run, collapsing all decision
        iterations.

        This is the unique items from calling `available_results`, with all decision iterations
        set to 0.

        This method is used to determine whether a model run is complete, given that it is
        impossible to know how many decision iterations to expect: we simply check that each
        expected timestep has been completed.

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        set Set of tuples representing available results
        """

        available_results = self.available_results(model_run_name)

        canonical_list = []

        for t, d, model_name, output_name in available_results:
            canonical_list.append((t, 0, model_name, output_name))

        # Return as a set to remove duplicates
        return set(canonical_list)

    def canonical_expected_results(self, model_run_name):
        """List the results that are expected from a model run, collapsing all decision
        iterations.

        For a complete model run, this would coincide with the unique list returned from
        `available_results`, where all decision iterations are set to 0.

        This method is used to determine whether a model run is complete, given that it is
        impossible to know how many decision iterations to expect: we simply check that each
        expected timestep has been completed.

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        set Set of tuples representing expected results
        """

        # Model results are returned as a tuple
        # (timestep, decision_it, model_name, output_name)
        # so we first build the full list of expected results tuples.

        expected_results = []

        # Get the sos model name given the model run name, and the full list of timesteps
        model_run = self.read_model_run(model_run_name)
        timesteps = sorted(model_run['timesteps'])
        sos_model_name = model_run['sos_model']

        # Get the list of sector models in the sos model
        sos_config = self.read_sos_model(sos_model_name)

        # For each sector model, get the outputs and create the tuples
        for model_name in sos_config['sector_models']:

            model_config = self.read_model(model_name)
            outputs = model_config['outputs']

            for output, t in itertools.product(outputs, timesteps):
                expected_results.append((t, 0, model_name, output['name']))

        # Return as a set to remove duplicates
        return set(expected_results)

    def canonical_missing_results(self, model_run_name):
        """List the results that are missing from a model run, collapsing all decision
        iterations.

        For a complete model run, this is what is left after removing
        canonical_available_results from canonical_expected_results.

        Parameters
        ----------
        model_run_name : str

        Returns
        -------
        set Set of tuples representing missing results
        """

        return self.canonical_expected_results(
            model_run_name) - self.canonical_available_results(model_run_name)

    def _get_result_darray_internal(self, model_run_name, model_name, output_name,
                                    time_decision_tuples):
        """Internal implementation for `get_result_darray`, after the unique list of
        (timestep, decision) tuples has been generated and validated.

        This method gets the spec for the output defined by the model_run_name, model_name
        and output_name and expands the spec to include an additional dimension for the list of
        tuples.

        Then, for each tuple, the data array from the corresponding read_results call is
        stacked, and together with the new spec this information is returned as a new
        DataArray.

        Parameters
        ----------
        model_run_name : str
        model_name : str
        output_name : str
        time_decision_tuples : list of unique (timestep, decision) tuples

        Returns
        -------
        DataArray with expanded spec and data for each (timestep, decision) tuple
        """

        # Get the output spec given the name of the sector model and output
        output_spec = None
        model = self.read_model(model_name)

        for output in model['outputs']:

            # Ignore if the output name doesn't match
            if output_name != output['name']:
                continue

            output_spec = Spec.from_dict(output)

        assert output_spec, "Output name was not found in model outputs"

        # Read the results for each (timestep, decision) tuple and stack them
        list_of_numpy_arrays = []

        for t, d in time_decision_tuples:
            d_array = self.read_results(model_run_name, model_name, output_spec, t, d)
            list_of_numpy_arrays.append(d_array.data)

        stacked_data = np.vstack(list_of_numpy_arrays)

        # Add new dimensions to the data spec
        output_dict = output_spec.as_dict()
        output_dict['dims'] = ['timestep_decision'] + output_dict['dims']
        output_dict['coords']['timestep_decision'] = time_decision_tuples

        output_spec = Spec.from_dict(output_dict)

        # Create a new DataArray from the modified spec and stacked data
        return DataArray(output_spec, np.reshape(stacked_data, output_spec.shape))

    def get_result_darray(self, model_run_name, model_name, output_name, timesteps=None,
                          decision_iterations=None, time_decision_tuples=None):
        """Return data for multiple timesteps and decision iterations for a given output from
        a given sector model in a specific model run.

        You can specify either:
            a list of (timestep, decision) tuples
                in which case data for all of those tuples matching the available results will
                be returned
        or:
            a list of timesteps
                in which case data for all of those timesteps (and any decision iterations)
                matching the available results will be returned
        or:
            a list of decision iterations
                in which case data for all of those decision iterations (and any timesteps)
                matching the available results will be returned
        or:
            a list of timesteps and a list of decision iterations
                in which case data for the Cartesian product of those timesteps and those
                decision iterations matching the available results will be returned
        or:
            nothing
                in which case all available results will be returned

        Then, for each tuple, the data array from the corresponding read_results call is
        stacked, and together with the new spec this information is returned as a new
        DataArray.

        Parameters
        ----------
        model_run_name : str
        model_name : str
        output_name : str
        timesteps : optional list of timesteps
        decision_iterations : optional list of decision iterations
        time_decision_tuples : optional list of unique (timestep, decision) tuples

        Returns
        -------
        DataArray with expanded spec and the data requested
        """
        available = self.available_results(model_run_name)

        # Build up the necessary list of tuples
        if not timesteps and not decision_iterations and not time_decision_tuples:
            list_of_tuples = [
                (t, d) for t, d, m, out in available
                if m == model_name and out == output_name
            ]

        elif timesteps and not decision_iterations and not time_decision_tuples:
            list_of_tuples = [
                (t, d) for t, d, m, out in available
                if m == model_name and out == output_name and t in timesteps
            ]

        elif decision_iterations and not timesteps and not time_decision_tuples:
            list_of_tuples = [
                (t, d) for t, d, m, out in available
                if m == model_name and out == output_name and d in decision_iterations
            ]

        elif time_decision_tuples and not timesteps and not decision_iterations:
            list_of_tuples = [
                (t, d) for t, d, m, out in available
                if m == model_name and out == output_name and (t, d) in time_decision_tuples
            ]

        elif timesteps and decision_iterations and not time_decision_tuples:
            t_d = list(itertools.product(timesteps, decision_iterations))
            list_of_tuples = [
                (t, d) for t, d, m, out in available
                if m == model_name and out == output_name and (t, d) in t_d
            ]

        else:
            msg = "Expected either timesteps, or decisions, or (timestep, decision) " + \
                  "tuples, or timesteps and decisions, or none of the above."
            raise ValueError(msg)

        if not list_of_tuples:
            raise SmifDataNotFoundError("None of the requested data is available.")

        return self._get_result_darray_internal(
            model_run_name, model_name, output_name, sorted(list_of_tuples)
        )

    def get_results(self,
                    model_run_names: list,
                    model_name: str,
                    output_names: list,
                    timesteps: list = None,
                    decisions: list = None,
                    time_decision_tuples: list = None,
                    ):
        """Return data for multiple timesteps and decision iterations for a given output from
        a given sector model for multiple model runs.

        Parameters
        ----------
        model_run_names: list[str]
            the requested model run names
        model_name: str
            the requested sector model name
        output_names: list[str]
            the requested output names (output specs must all match)
        timesteps: list[int]
            the requested timesteps
        decisions: list[int]
            the requested decision iterations
        time_decision_tuples: list[tuple]
            a list of requested (timestep, decision) tuples

        Returns
        -------
        dict
            Nested dictionary of DataArray objects, keyed on model run name and output name.
            Returned DataArrays include one extra (timestep, decision_iteration) dimension.
        """

        # List the available output names and verify requested outputs match
        outputs = self.read_model(model_name)['outputs']
        available_outputs = [output['name'] for output in outputs]

        for output_name in output_names:
            assert output_name in available_outputs, \
                '{} is not an output of sector model {}.'.format(output_name, model_name)

        # The spec for each requested output must be the same. We check they have the same
        # coordinates
        coords = [Spec.from_dict(output).coords for output in outputs if
                  output['name'] in output_names]

        for coord in coords:
            if coord != coords[0]:
                raise ValueError('Different outputs must have the same coordinates')

        # Now actually obtain the requested results
        results_dict = OrderedDict()  # type: OrderedDict
        for model_run_name in model_run_names:
            results_dict[model_run_name] = OrderedDict()
            for output_name in output_names:
                results_dict[model_run_name][output_name] = self.get_result_darray(
                    model_run_name,
                    model_name,
                    output_name,
                    timesteps,
                    decisions,
                    time_decision_tuples
                )
        return results_dict

    # endregion

    # region data store utilities
    def _key_from_data(self, path, *args):
        """Return path or generate a unique key for a given set of args
        """
        if isinstance(self.data_store, (CSVDataStore, ParquetDataStore)):
            return path
        else:
            return tuple(args)
    # endregion


def _pick_from_list(list_of_dicts, name):
    for item in list_of_dicts:
        if 'name' in item and item['name'] == name:
            return item
    return None
