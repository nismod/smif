# -*- coding: utf-8 -*-
"""Read and parse the config for sector models
"""
import logging
import os
from glob import glob

import fiona

from .load import load
from .validate import validate_interventions


class SectorModelReader(object):
    """Parses the configuration and input data for a sector model

    Arguments
    =========
    initial_config : dict
        Sector model details, sufficient to read the full config from a set of
        files. Must contain the following fields:

            "model_name"
                The name of the sector model, for reference within the
                system-of-systems model

            "model_path"
                The path to the python module file that contains an
                implementation of SectorModel

            "model_classname"
                The name of the class that implements SectorModel

            "model_config_dir"
                The root path of model config/data to use, which
                must contain inputs.yaml, outputs.yaml, time_intervals.yaml and
                regions.shp/regions.geojson

            "initial_conditions"
                List of files containing initial conditions

            "interventions"
                List of files containing interventions

    """
    def __init__(self, initial_config=None):
        self.logger = logging.getLogger(__name__)

        if initial_config is not None:
            self.model_name = initial_config["model_name"]
            self.model_path = initial_config["model_path"]
            self.model_classname = initial_config["model_classname"]
            self.model_config_dir = initial_config["model_config_dir"]
            self.initial_conditions_paths = initial_config["initial_conditions"]
            self.interventions_paths = initial_config["interventions"]
        else:
            self.model_name = None
            self.model_path = None
            self.model_classname = None
            self.model_config_dir = None
            self.initial_conditions_paths = None
            self.interventions_paths = None

        self.inputs = None
        self.outputs = None
        self.time_intervals = None
        self.regions = None

        self.initial_conditions = None
        self.interventions = None

    def load(self):
        """Load and check all config
        """
        self.inputs = self.load_inputs()
        self.outputs = self.load_outputs()
        self.time_intervals = self.load_time_intervals()
        self.regions = self.load_regions()
        self.initial_conditions = self.load_initial_conditions()
        self.interventions = self.load_interventions()

    @property
    def data(self):
        """Expose all loaded config data

        Returns
        =======
        data : dict
            Model configuration data, with the following fields:
                "name": The name of the sector model, for reference within the
                system-of-systems model

                "path": The path to the python module file that contains an
                implementation of SectorModel

                "classname": The name of the class that implements SectorModel

                "inputs": A list of the inputs that this model requires

                "outputs": A list of the outputs that this model provides

                "time_intervals": A list of time intervals within a year that are
                represented by the model, each with reference to the model's
                internal identifier for timesteps

                "regions": A list of geographical regions used within the model, as
                objects with both geography and attributes

                "initial_conditions": A list of initial conditions required to set up
                the modelled system in the base year

                "interventions": A list of possible interventions that could be made
                in the modelled system

        """
        return {
            "name": self.model_name,
            "path": self.model_path,
            "classname": self.model_classname,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "time_intervals": self.time_intervals,
            "regions": self.regions,
            "initial_conditions": self.initial_conditions,
            "interventions": self.interventions
        }

    def load_inputs(self):
        """Input spec is located in the ``data/<sectormodel>/inputs.yaml`` file

        """
        path = os.path.join(self.model_config_dir, 'inputs.yaml')

        if not os.path.exists(path):
            msg = "inputs config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))

        return load(path)

    def load_outputs(self):
        """Output spec is located in ``data/<sectormodel>/output.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'outputs.yaml')

        if not os.path.exists(path):
            msg = "outputs config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))

        return load(path)

    def load_initial_conditions(self):
        """Inital conditions are located in yaml files
        specified in sector model blocks in the sos model config
        """
        data = []

        paths = self.initial_conditions_paths
        for path in paths:
            self.logger.debug("Loading initial conditions from %s", path)
            new_data = load(path)
            data.extend(new_data)
        return data

    def load_interventions(self):
        """Interventions are located in yaml files
        specified in sector model blocks in the sos model config
        """
        data = []
        paths = self.interventions_paths
        for path in paths:
            self.logger.debug("Loading interventions from %s", path)
            new_data = load(path)
            validate_interventions(new_data, path)
            data.extend(new_data)
        return data

    def load_time_intervals(self):
        """Within-year time intervals are specified in ``data/<sectormodel>/time_intervals.yaml``

        These specify the mapping of model timesteps to durations within a year
        (assume modelling 365 days: no extra day in leap years, no leap seconds)

        Each time interval must have
        - start (period since beginning of year)
        - end (period since beginning of year)
        - id (label to use when passing between integration layer and sector model)

        use ISO 8601[1]_ duration format to specify periods::

            P[n]Y[n]M[n]DT[n]H[n]M[n]S

        For example::

            P1Y == 1 year
            P3M == 3 months
            PT168H == 168 hours

        So to specify a period from the beginning of March to the end of May::

            start: P2M
            end: P5M
            id: spring

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations

        """
        path = os.path.join(self.model_config_dir, 'time_intervals.yaml')

        if not os.path.exists(path):
            msg = "time_intervals config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))

        return load(path)

    def load_regions(self):
        """Model regions are specified in ``data/<sectormodel>/regions.*``

        The file format must be possible to parse with GDAL, and must contain
        an attribute "name" to use as an identifier for the region.
        """
        path = os.path.join(self.model_config_dir, 'regions.shp')

        if not os.path.exists(path):
            paths = glob("{}/regions.*".format(self.model_config_dir))
            if len(paths) == 1:
                path = paths[0]
            else:
                msg = "regions config file not found for {} model"
                raise FileNotFoundError(msg.format(self.model_name))

        with fiona.drivers():
            with fiona.open(path) as src:
                data = [f for f in src]

        return data
