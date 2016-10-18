# Parse yaml config files, to construct sector models
import yaml
import jsonschema

class ConfigParser:
    """Parse yaml config file,
    hold config data,
    validate config data against required set
    """
    def __init__(self, filepath):
        self._config_filepath = filepath

        with open(filepath, 'r') as fh:
            self._config_data = yaml.load(fh)

    def data(self):
        return self._config_data

    def validate(self,schema):
        if self._config_data is None:
            raise AttributeError("Config data not loaded")

        jsonschema.validate(self._config_data, schema)
