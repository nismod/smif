# -*- coding: utf-8 -*-
"""Read and parse the config for sector models
"""
import os
from glob import glob
import fiona
from . parse_config import ConfigParser


class SectorModelReader(object):
    """Parses the configuration and input data for a sector model

    Arguments
    =========
    model_name : str
        The name of the model
    model_path : str
        The path to the python module file that contains an implementation
        of SectorModel
    model_classname : str
        The name of the class that implements SectorModel
    model_config_dir : str
        The root path of model config/data to use

    """
    def __init__(self, model_name, model_path, model_classname,
                 model_config_dir):
        self.model_name = model_name
        self.model_path = model_path
        self.model_classname = model_classname
        self.model_config_dir = model_config_dir

        self.inputs = None
        self.outputs = None
        self.time_intervals = None
        self.regions = None

        self.assets = None
        self.interventions = None

    def load(self):
        """Load and check all config
        """
        self.inputs = self._load_inputs()
        self.outputs = self._load_outputs()
        self.time_intervals = self._load_time_intervals()
        self.regions = self._load_regions()
        self.assets = self._load_assets()
        self.interventions = self._load_interventions()

    @property
    def data(self):
        """Expose all loaded config data
        """
        return {
            "name": self.model_name,
            "path": self.model_path,
            "classname": self.model_classname,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "time_intervals": self.time_intervals,
            "regions": self.regions,
            "assets": self.assets,
            "interventions": self.interventions
        }

    def _load_inputs(self):
        """Input spec is located in the ``data/<sectormodel>/inputs.yaml`` file

        """
        path = os.path.join(self.model_config_dir, 'inputs.yaml')

        if not os.path.exists(path):
            msg = "inputs config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))

        return ConfigParser(path).data

    def _load_outputs(self):
        """Output spec is located in ``data/<sectormodel>/output.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'outputs.yaml')

        if not os.path.exists(path):
            msg = "outputs config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))

        return ConfigParser(path).data

    def _load_assets(self):
        """Assets are located in ``data/<sectormodel>/assets/*.yaml`` files
        """
        data = []

        paths = glob("{}/assets/*.yaml".format(self.model_config_dir))
        if len(paths) == 0:
            msg = "Assets config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))
        else:
            for path in paths:
                new_data = ConfigParser(path).data
                data.extend(new_data)
        return data

    def _load_interventions(self):
        """Interventions are in ``data/<sectormodel>/interventions/*.yaml`` files
        """
        data = []
        paths = glob("{}/interventions/*.yaml".format(self.model_config_dir))
        if len(paths) == 0:
            msg = "Interventions config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))
        else:
            for path in paths:
                new_data = ConfigParser(path).data
                data.extend(new_data)
        return data

    def _load_time_intervals(self):
        """Within-year time intervals are specified in ``data/<sectormodel>/time_intervals.yaml``

        These specify the mapping of model timesteps to durations within a year
        (assume modelling 365 days: no extra day in leap years, no leap seconds)

        Each time interval must have
        - start (period since beginning of year)
        - end (period since beginning of year)
        - id (label to use when passing between integration layer and sector model)

        use ISO 8601[1]_ duration format to specify periods::

            P[n]Y[n]M[n]DT[n]H[n]M[n]S

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/ISO_8601#Durations

        """
        path = os.path.join(self.model_config_dir, 'time_intervals.yaml')

        if not os.path.exists(path):
            msg = "time_intervals config file not found for {} model"
            raise FileNotFoundError(msg.format(self.model_name))

        return ConfigParser(path).data

    def _load_regions(self):
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
