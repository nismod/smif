# Parse yaml config files, to construct sector models
import yaml
import jsonschema

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
