"""Results provides a common interface to access results from model runs.
"""

from typing import Union

import pandas as pd
from smif.data_layer.store import Store


class Results:
    """Common interface to access results from model runs.

    Parameters
    ----------
    store: Store or dict
        pre-created Store object or dictionary of the form {'interface': <interface>,
        'dir': <dir>} where <interface> is either 'local_csv' or 'local_parquet', and <dir> is
        the model base directory
    """
    def __init__(self, store: Union[Store, dict]):

        if type(store) is dict:
            self._store = Store.from_dict(store)
        else:
            self._store = store  # type: Store

        # keep tabs on the units of any read outputs
        self._output_units = dict()  # type: dict

    def list_model_runs(self):
        """Return a list of model run names.

        Returns
        -------
        List of model run names
        """
        return sorted([x['name'] for x in self._store.read_model_runs()])

    def available_results(self, model_run_name):
        """Return the results available for a given model run.

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
             model_names: list,
             output_names: list,
             timesteps: list = None,
             decisions: list = None,
             time_decision_tuples: list = None,
             ):
        """Return results from the store as a formatted pandas data frame. There are a number
        of ways of requesting specific timesteps/decisions. You can specify either:

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

        Parameters
        ----------
        model_run_names: list
            the requested model run names
        model_names: list
            the requested sector model names (exactly one required)
        output_names: list
            the requested output names (output specs must all match)
        timesteps: list
            the requested timesteps
        decisions: list
            the requested decision iterations
        time_decision_tuples: list
            a list of requested (timestep, decision) tuples

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

        Returns
        -------
        pandas.DataFrame
        """

        self.validate_names(model_run_names, model_names, output_names)

        results_dict = self._store.get_results(
            model_run_names,
            model_names[0],
            output_names,
            timesteps,
            decisions,
            time_decision_tuples
        )

        # Keep tabs on the units for each output
        for model_run_name in model_run_names:
            for output_name in output_names:
                res = results_dict[model_run_name][output_name]
                self._output_units[res.name] = res.unit

        # For each output, concatenate all requested model runs into a single data frame
        formatted_frames = []
        for output_name in output_names:
            # Get each DataArray as a pandas data frame and concatenate, resetting the index to
            # give back a flat data array
            list_of_df = [results_dict[x][output_name].as_df() for x in model_run_names]
            names_of_df = [x for x in results_dict.keys()]

            formatted_frames.append(
                pd.concat(list_of_df, keys=names_of_df, names=['model_run']).reset_index())

        # Append the other output columns to the first data frame
        formatted_frame = formatted_frames.pop(0)
        output_names.pop(0)

        for other_frame, output_name in zip(formatted_frames, output_names):
            assert (formatted_frame['model_run'] == other_frame['model_run']).all()
            assert (formatted_frame['timestep_decision'] == other_frame[
                'timestep_decision']).all()
            formatted_frame[output_name] = other_frame[output_name]

        # Unpack the timestep_decision tuples into individual columns and drop the combined
        formatted_frame[['timestep', 'decision']] = pd.DataFrame(
            formatted_frame['timestep_decision'].tolist(), index=formatted_frame.index)

        formatted_frame = formatted_frame.drop(columns=['timestep_decision'])

        # Now reorder the columns. Want model_run then timestep then decision
        cols = formatted_frame.columns.tolist()

        assert (cols[0] == 'model_run')
        cols.insert(1, cols.pop(cols.index('timestep')))
        cols.insert(2, cols.pop(cols.index('decision')))
        assert (cols[0:3] == ['model_run', 'timestep', 'decision'])

        return formatted_frame[cols]

    def get_units(self, output_name: str):
        """ Return the units of a given output.

        Parameters
        ----------
        output_name: str

        Returns
        -------
        str
        """
        return self._output_units[output_name]

    def validate_names(self, model_run_names, sec_model_names, output_names):

        if len(sec_model_names) != 1:
            raise NotImplementedError(
                'Results.read() currently requires exactly one sector model'
            )

        if len(model_run_names) < 1:
            raise ValueError(
                'Results.read() requires at least one sector model name'
            )

        if len(output_names) < 1:
            raise ValueError(
                'Results.read() requires at least one output name'
            )
