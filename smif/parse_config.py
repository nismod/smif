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

        jsonschema.validate(self.data, schema)

    def validate_as_modelrun_config(self):
        """Validate the loaded data as required for model run configuration
        """
        model_config_schema_path = os.path.join(os.path.dirname(__file__), "schema", "modelrun_config_schema.json")
        with open(model_config_schema_path, 'r') as fh:
            schema = json.load(fh)

        self.validate(schema)
