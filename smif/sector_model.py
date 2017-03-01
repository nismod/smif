# -*- coding: utf-8 -*-
"""This module acts as a bridge to the sector models from the controller

The :class:`SectorModel` exposes several key methods for running wrapped
sector models.  To add a sector model to an instance of the framework,
first implement :class:`SectorModel`.

Key Functions
=============
This class performs several key functions which ease the integration of sector
models into the system-of-systems framework.

The user must implement the various abstract functions throughout the class to
provide an interface to the sector model, which can be called upon by the
framework. From the model's perspective, :class:`SectorModel` provides a bridge
from the sector-specific problem representation to the general representation
which allows reasoning across infrastructure systems.

The key functions include

* converting input/outputs to/from geographies/temporal resolutions
* converting control vectors from the decision layer of the framework, to
  asset Interventions specific to the sector model
* returning scaler/vector values to the framework to enable measurements of
  performance, particularly for the purposes of optimisation and rule-based
  approaches


"""
import importlib
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize
from smif.inputs import ModelInputs
from smif.outputs import ModelOutputs

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"


class SectorModel(ABC):
    """A representation of the sector model with inputs and outputs

    """
    def __init__(self):
        self._model_name = None
        self._schema = None

        self.interventions = []

        self._inputs = ModelInputs({})
        self._outputs = ModelOutputs({})

        self.logger = logging.getLogger(__name__)

    def validate(self):
        """Validate that this SectorModel has been set up with sufficient data
        to run
        """
        pass

    @property
    def name(self):
        """The name of the sector model

        Returns
        =======
        str
            The name of the sector model

        Note
        ====
        The name corresponds to the name of the folder in which the
        configuration is expected to be found

        """
        return self._model_name

    @name.setter
    def name(self, value):
        self._model_name = value

    @property
    def inputs(self):
        """The inputs to the model

        Returns
        =======
        :class:`smif.inputs.ModelInputs`

        """
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """The inputs/dependencies to the model

        The inputs should be specified in a dictionary.  For example::

                dependencies:
                - name: eletricity_price
                spatial_resolution: GB
                temporal_resolution: annual
                from_model: energy_supply

        Arguments
        =========
        value : dict
            A dictionary of inputs to the model. This may include parameters,
            assets and exogenous data.

        """
        assert isinstance(value, dict)
        self._inputs = ModelInputs(value)

    @property
    def outputs(self):
        """The outputs from the model

        Arguments
        =========
        value : dict
            A dictionary of outputs from the model. This may include results
            and metrics

        Returns
        =======
        :class:`smif.outputs.ModelOutputs`

        """
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = ModelOutputs(value)

    @property
    def intervention_names(self):
        """The names of the interventions

        Returns
        =======
        list
            A list of the names of the interventions
        """
        return [intervention['name'] for intervention in self.interventions]


    @abstractmethod
    def simulate(self, decisions, state, data):
        """Implement this method to run the model

        Arguments
        ---------
        decisions: list
        state: list
        data: dict

        Returns
        -------
        dict
            A nested dictionary of the format:
            ``results[timestep][parameter][region][time_interval] = {value, units}``

        Notes
        -----
        In the results returned from the :py:meth:`simulate` method:

        ``time_interval``
            should reference the id from the time_interval
            specification in the ``time_intervals.yaml`` file
        ``region``
            should reference the region name from the ``regions.*`` geographies
            files specified in the configuration
        ``timestep``
            refers to the current year, and can always be derived from
            ``data['timestep']`` which is passed by `smif` to the :py:meth:`simulate` method.

        """
        pass

    @abstractmethod
    def extract_obj(self, results):
        """Implement this method to return a scalar value objective function

        This method should take the results from the output of the
        :py:meth:`simulate` method, process the results,
        and return a scalar value which can be used as a component of
        the objective function by the decision layer

        Arguments
        =========
        results : dict
            A nested dict of the results from the :py:meth:`simulate` method

        Returns
        =======
        float
            A scalar component generated from the simulation model results
        """
        pass
 

    ##
    # File interaction methods like load_results_from_files and
    # replace_{line,cell} might live somewhere else
    # - maybe sectormodel, if that deals with replacing file data?
    # - maybe a utility file handler class?
    # seem related to outputs for now
    ##
    @staticmethod
    def replace_line(file_name, line_num, new_data):
        """Replaces a line in a file with new data

        Arguments
        =========
        file_name: str
            The path to the input file
        line_num: int
            The number of the line to replace
        new_data: str
            The data to replace in the line

        """
        lines = open(file_name, 'r').readlines()
        lines[line_num] = new_data
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()

    @staticmethod
    def replace_cell(file_name, line_num, column_num, new_data,
                     delimiter=None):
        """Replaces a cell in a delimited file with new data

        Arguments
        =========
        file_name: str
            The path to the input file
        line_num: int
            The number of the line to replace (0-index)
        column_num: int
            The number of the column to replace (0-index)
        new_data: str
            The data to replace in the line
        delimiter: str, default=','
            The delimiter of the columns
        """
        line_num -= 1
        column_num -= 1

        with open(file_name, 'r') as input_file:
            lines = input_file.readlines()

        columns = lines[line_num].split(delimiter)
        columns[column_num] = new_data
        lines[line_num] = " ".join(columns) + "\n"

        with open(file_name, 'w') as out_file:
            out_file.writelines(lines)

    @property
    def file_locations(self):
        """Allows ordered searching through file to extract results from file
        """
        locations = {}

        # For each file return tuple of row/column
        for name in self.names:
            row_num = self.values[name]['row_num']
            col_num = self.values[name]['col_num']
            filename = self.values[name]['file_name']

            if filename not in locations:
                # add filename to locations
                locations[filename] = {}

            locations[filename][name] = (row_num, col_num)

        return locations

    def load_results_from_files(self, dirname):
        """Load model outputs from specified files
        """
        for filename in self.file_locations:
            file_path = os.path.join(dirname, filename)

            with open(file_path, 'r') as results_set:
                lines = results_set.readlines()

            for name, rowcol in self.file_locations[filename].items():
                row_num, col_num = rowcol
                result = lines[row_num][col_num:].strip()
                self.values[name]['value'] = result


class SectorModelBuilder(object):
    """Build the components that make up a sectormodel from the configuration

    Parameters
    ----------
    name : str
        The name of the sector model

    """

    def __init__(self, name):
        self._sector_model_name = name
        self._sector_model = None
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path, classname):
        """Dynamically load model module

        """
        if os.path.exists(model_path):
            self.logger.info("Importing run module from %s", model_path)

            spec = importlib.util.spec_from_file_location(
                self._sector_model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            klass = module.__dict__[classname]

            self._sector_model = klass()
            self._sector_model.name = self._sector_model_name

        else:
            msg = "Cannot find {} for the {} model".format(
                model_path, self._sector_model_name)
            raise Exception(msg)

    def add_inputs(self, input_dict):
        """Add inputs to the sector model
        """
        msg = "Sector model must be loaded before adding inputs"
        assert self._sector_model is not None, msg

        self._sector_model.inputs = input_dict

    def add_outputs(self, output_dict):
        """Add outputs to the sector model
        """
        msg = "Sector model must be loaded before adding outputs"
        assert self._sector_model is not None, msg

        self._sector_model.outputs = output_dict

    def add_assets(self, asset_list):
        """Add assets to the sector model

        Parameters
        ----------
        asset_list : list
            A list of dicts of assets
        """
        msg = "Sector model must be loaded before adding assets"
        assert self._sector_model is not None, msg

        self._sector_model.assets = asset_list

    def add_interventions(self, intervention_list):
        """Add interventions to the sector model

        Parameters
        ----------
        intervention_list : list
            A list of dicts of interventions

        """
        msg = "Sector model must be loaded before adding interventions"
        assert self._sector_model is not None, msg

        self._sector_model.interventions = intervention_list

    def validate(self):
        """Check and/or assert that the sector model is correctly set up
        - should raise errors if invalid
        """
        assert self._sector_model is not None, "Sector model not loaded"
        self._sector_model.validate()
        return True

    def finish(self):
        """Validate and return the sector model
        """
        self.validate()
        return self._sector_model
