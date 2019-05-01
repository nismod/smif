"""Results provides a common interface to access results from model runs.

Raises
------
SmifDataNotFoundError
    If data cannot be found in the store when try to read from the store
SmifDataMismatchError
    Data presented to read, write and update methods is in the
    incorrect format or of wrong dimensions to that expected
SmifDataReadError
    When unable to read data e.g. unable to handle file type or connect
    to database
"""

import os

import pandas as pd
from smif.data_layer.file import (CSVDataStore, FileMetadataStore,
                                  ParquetDataStore, YamlConfigStore)
from smif.data_layer.store import Store


class Results:
    """Common interface to access results from model runs.

    Parameters
    ----------
    details_dict: dict optional dictionary of the form {'interface': <interface>, 'dir': <dir>}
        where <interface> is either 'local_csv' or 'local_parquet', and <dir> is the model base
        directory
    store: Store optional pre-created Store object
    """

    def __init__(self, details_dict: dict = None, store: Store = None):

        assert bool(details_dict) != bool(store), \
            'Results() accepts either a details dict or a store'

        self._store = store
        if store:
            return

        try:
            interface = details_dict['interface']
        except KeyError:
            print('No interface provided for Results().  Assuming local_csv.')
            interface = 'local_csv'

        try:
            directory = details_dict['dir']
        except KeyError:
            print('No directory provided for Results().  Assuming \'.\'.')
            directory = '.'

        # Check that the provided interface is supported
        file_store = self._get_file_store(interface)
        if file_store is None:
            raise ValueError(
                'Unsupported interface "{}". Supply local_csv or local_parquet'.format(
                    interface))

        # Check that the directory is valid
        if not os.path.isdir(directory):
            raise ValueError('Expected {} to be a valid directory'.format(directory))

        self._store = Store(
            config_store=YamlConfigStore(directory),
            metadata_store=FileMetadataStore(directory),
            data_store=file_store(directory),
            model_base_folder=directory
        )

        # Create an empty dictionary for keeping tabs on the units of any read outputs
        self._output_units = dict()

    @staticmethod
    def _get_file_store(interface):
        """ Return the appropriate derived FileDataStore class, or None if the requested
        interface is invalid.

        Parameters
        ----------
        interface: str the requested interface

        Returns
        -------
        The appropriate derived FileDataStore class
        """
        return {
            'local_csv': CSVDataStore,
            'local_parquet': ParquetDataStore,
        }.get(interface, None)

    def list_model_runs(self):
        """ Return a list of model run names.

        Returns
        -------
        List of model run names
        """
        return sorted([x['name'] for x in self._store.read_model_runs()])

    def available_results(self, model_run_name):
        """ Return the results available for a given model run.

        Parameters
        ----------
        model_run_name: str the requested model run

        Returns
        -------
        A nested dictionary data structure of the results available for the given model run
        """

        available = self._store.available_results(model_run_name)

        results = {
            'model_run': model_run_name,
            'sos_model': self._store.read_model_run(model_run_name)['sos_model'],
            'sector_models': dict(),
        }

        model_names = {sec for _t, _d, sec, _out in available}
        for model_name in model_names:
            results['sector_models'][model_name] = {
                'outputs': dict(),
            }

            outputs = {out for _t, _d, sec, out in available if sec == model_name}

            for output in outputs:
                results['sector_models'][model_name]['outputs'][output] = dict()

                decs = {d for _t, d, sec, out in available if
                        sec == model_name and out == output}

                for dec in decs:
                    ts = sorted({t for t, d, sec, out in available if
                                 d == dec and sec == model_name and out == output})
                    results['sector_models'][model_name]['outputs'][output][dec] = ts

        return results

    def read(self,
             model_run_names: list,
             sec_model_names: list,
             output_names: list,
             timesteps: list = None,
             decisions: list = None,
             time_decision_tuples: list = None,
             ):
        """ Return the results from the store.

        Parameters
        ----------
        model_run_names: list the requested model run names
        sec_model_names: list the requested sector model names (exactly one required)
        output_names: list the requested output names (exactly one required)
        timesteps: list the requested timesteps
        decisions: list the requested decision iterations
        time_decision_tuples: list a list of requested (timestep, decision) tuples

        Returns
        -------
        A Pandas dataframe
        """

        if len(sec_model_names) != 1:
            raise NotImplementedError(
                'Results.read() currently requires exactly one sector model'
            )

        if len(output_names) != 1:
            raise NotImplementedError(
                'Results.read() currently requires exactly one output'
            )

        results_dict = self._store.get_results(
            model_run_names,
            sec_model_names[0],
            output_names[0],
            timesteps,
            decisions,
            time_decision_tuples
        )

        # Keep tabs on the units for each output
        for x in results_dict.values():
            self._output_units[x.name] = x.unit

        # Get each DataArray as a pandas data frame and concatenate, resetting the index to
        # give back a flat data array
        list_of_df = [x.as_df() for x in results_dict.values()]
        names_of_df = [x for x in results_dict.keys()]

        results = pd.concat(list_of_df, keys=names_of_df, names=['model_run']).reset_index()

        # Unpack the timestep_decision tuples into individual columns and return
        results[['timestep', 'decision']] = pd.DataFrame(results['timestep_decision'].tolist(),
                                                         index=results.index)

        return results.drop(columns=['timestep_decision'])

        # Rename the output columns to include units
        renamed_cols = dict()
        for key, val in self._output_units.items():
            renamed_cols[key] = '{}_({})'.format(key, val)
        results = results.rename(index=str, columns=renamed_cols)

    def get_units(self, output_name: str):
        """ Return the units of a given output.

        Parameters
        ----------
        output_name: the name of the output

        Returns
        -------
        str the units of the output
        """
        return self._output_units[output_name]
