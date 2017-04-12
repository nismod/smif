# -*- coding: utf-8 -*-
"""Read and parse the config for sector models
"""
import logging
import os

from .load import load
from .validate import (validate_initial_conditions, validate_input_spec,
                       validate_interventions, validate_output_spec)


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
        self.initial_conditions = None
        self.interventions = None

    def load(self):
        """Load and check all config
        """
        self.inputs = self.load_inputs()
        self.outputs = self.load_outputs()
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
            "initial_conditions": self.initial_conditions,
            "interventions": self.interventions
        }

    def load_inputs(self):
        """Input spec is located in the ``data/<sectormodel>/inputs.yaml`` file

        """
        return self.load_io_metadata('inputs')

    def load_outputs(self):
        """Output spec is located in ``data/<sectormodel>/output.yaml`` file
        """
        return self.load_io_metadata('outputs')

    def load_io_metadata(self, inputs_or_outputs):
        """Load inputs or outputs, allowing missing or empty file
        """
        path = os.path.join(
            self.model_config_dir,
            '{}.yaml'.format(inputs_or_outputs))

        if not os.path.exists(path):
            msg = "No %s provided for '%s' model: %s not found"
            self.logger.warning(msg, inputs_or_outputs, self.model_name, path)
            data = []

        else:
            file_contents = load(path)

            if file_contents is None:
                self.logger.warning(
                    "No %s provided for '%s' model: %s was empty",
                    inputs_or_outputs,
                    self.model_name,
                    path)
                data = []
            else:
                data = file_contents

        if inputs_or_outputs == 'inputs':
            validate_input_spec(data, self.model_name)
        else:
            validate_output_spec(data, self.model_name)

        return data

    def load_initial_conditions(self):
        """Inital conditions are located in yaml files
        specified in sector model blocks in the sos model config
        """
        data = []
        for path in self.initial_conditions_paths:
            self.logger.debug("Loading initial conditions from %s", path)
            new_data = load(path)
            validate_initial_conditions(new_data, path)
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
