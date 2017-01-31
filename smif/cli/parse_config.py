# -*- coding: utf-8 -*-
"""Parse yaml config files, to construct sector models
"""
import json
import os

import jsonschema
import yaml


class ConfigParser:
    """Parse yaml config file,
    hold config data,
    validate config data against required set
    """
    def __init__(self, filepath=None):
        if filepath is not None:
            self._config_filepath = filepath

            with open(filepath, 'r') as fh:
                self.data = yaml.load(fh)
        else:
            self.data = None

    def validate(self, schema):
        """Validate data against a schema dict
        """
        if self.data is None:
            raise AttributeError("Config data not loaded")

        self._validate_against_schema(self.data, schema)

    def _validate_against_schema_file(self, data, schema_filename):
        """Validate data against a schema file
        """
        schema_filepath = self._get_schema_filepath(schema_filename)
        schema = self._load_schema_from_file(schema_filepath)
        self._validate_against_schema(data, schema)

    @staticmethod
    def _get_schema_filepath(schema_filename):
        return os.path.join(os.path.dirname(__file__),
                            "..",
                            "schema",
                            schema_filename)

    @staticmethod
    def _load_schema_from_file(schema_filename):
        with open(schema_filename, 'r') as file_handle:
            schema = json.load(file_handle)
        return schema

    @staticmethod
    def _validate_against_schema(data, schema):
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as error:
            raise ValueError(error.message)

    def validate_as_modelrun_config(self):
        """Validate the loaded data as required for model run configuration
        """
        if self.data is None:
            raise AttributeError("Config data not loaded")

        self._validate_against_schema_file(self.data,
                                           "modelrun_config_schema.json")

        for planning_type in self.data["planning"].values():
            if planning_type["use"] and "files" not in planning_type:
                msg = "A planning type needs files if it is going to be used."
                raise ValueError(msg)

    def validate_as_timesteps(self):
        """Validate the loaded data as required for model run timesteps
        """
        self._validate_against_schema_file(self.data,
                                           "timesteps_config_schema.json")

    def validate_as_assets(self):
        """Validate the loaded data as required for model run assets
        """
        self._validate_against_schema_file(self.data,
                                           "assets_schema.json")

        # except for some keys which are allowed simple values,
        simple_keys = ["type", "sector", "location"]
        # expect each attribute to be of the form {value: x, units: y}
        for asset in self.data:
            for key, value in asset.items():
                if key not in simple_keys and (
                        not isinstance(value, dict)
                        or "value" not in value
                        or "units" not in value
                    ):
                    fmt = "{0}.{1} was {2} but should have specified units, " + \
                          "e.g. {{'value': {2}, 'units': 'm'}}"

                    msg = fmt.format(asset["type"], key, value)
                    raise ValueError(msg)

    def validate_as_pre_specified_planning(self):
        """Validate the loaded data as a pre-specified planning file
        """
        self._validate_against_schema_file(self.data,
                                           "pre_specified_schema.json")
