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

from smif.data_layer.file import (CSVDataStore, FileMetadataStore,
                                  ParquetDataStore, YamlConfigStore)
from smif.data_layer.store import Store


class Results:
    """Common interface to access results from model runs.

    Parameters
    ----------
    interface: str the requested interface (local_csv or local_parquet currently supported)
    model_base_dir: str the base directory of the model
    """
    def __init__(self, interface='local_csv', model_base_dir='.'):

        # Check that the provided interface is supported
        file_store = self._get_file_store(interface)
        if file_store is None:
            raise ValueError(
                'Unsupported interface "{}". Supply local_csv or local_parquet'.format(
                    interface))

        # Check that the directory is valid
        if not os.path.isdir(model_base_dir):
            raise ValueError('Expected {} to be a valid directory'.format(model_base_dir))

        self._store = Store(
            config_store=YamlConfigStore(model_base_dir),
            metadata_store=FileMetadataStore(model_base_dir),
            data_store=file_store(model_base_dir),
            model_base_folder=model_base_dir
        )

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
