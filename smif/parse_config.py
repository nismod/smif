# Parse yaml config files, to construct sector models
import json
import jsonschema
import os
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

    def validate(self,schema):
        if self.data is None:
            raise AttributeError("Config data not loaded")

        self.validate_against_schema(self.data, schema)


    def _validate_against_schema_file(self, data, schema_filename):
        """Validate data against a schema file
        """
        schema_filepath = self._get_schema_filepath(schema_filename)
        schema = self._load_schema_from_file(schema_filepath)
        self.validate_against_schema(data, schema)

    def _get_schema_filepath(self, schema_filename):
        return os.path.join(os.path.dirname(__file__), "schema", schema_filename)

    def _load_schema_from_file(self,schema_filename):
        with open(schema_filename, 'r') as fh:
            schema = json.load(fh)
        return schema

    def validate_against_schema(self, data, schema):
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(e.message)

    def validate_as_modelrun_config(self):
        """Validate the loaded data as required for model run configuration
        """
        if self.data is None:
            raise AttributeError("Config data not loaded")

        self._validate_against_schema_file(self.data, "modelrun_config_schema.json")

        for planning_type in self.data["planning"].values():
            if planning_type["use"] and "files" not in planning_type:
                raise ValueError("A planning type needs files if it is going to be used.")

    def validate_as_timesteps(self):
        self._validate_against_schema_file(self.data, "timesteps_config_schema.json")
