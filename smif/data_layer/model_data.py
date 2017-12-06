"""Provide an interface to data, parameters and results

A`DataHandle` is passed in to a Model at runtime, to provide transparent
access to the relevant data and parameters for the current `ModelRun` and
iteration. It gives read access to parameters and input data (at any computed or
pre-computed timestep) and write access to output data (at the current
timestep).
"""


class DataHandle(object):
    """Get/set model parameters and data
    """
    def __init__(self, store, modelrun_id, iteration_id=None):
        """Create a DataHandle

        Parameters
        ----------
        store : DataInterface or dict
            The backing store for inputs, parameters, results
        modelrun_id : str or int
            The id of the current modelrun
        iteration_id :
            The id of the current ModelSet iteration
        """
        self._store = store
        self._modelrun_id = modelrun_id
        self._iteration_id = iteration_id

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def get_data(self, input_name):
        """Get data required for model inputs

        Parameters
        ----------
        input_name : str
        """
        return self._store[input_name]

    def get_parameter(self, parameter_name):
        """Get parameter values

        Parameters
        ----------
        parameter_name : str
        """
        return self._store[parameter_name]

    def set_results(self, output_name, data):
        """Set results values for model outputs

        Parameters
        ----------
        output_name : str
        data : numpy.ndarray
        """
        self._store[output_name] = data
