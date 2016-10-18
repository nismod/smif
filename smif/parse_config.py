# Parse yaml config files, to construct sector models
import yaml

class ConfigParser:
    def __init__(self, filepath):
        self._config_filepath = filepath

        with open(filepath, 'r') as fh:
            self._config_data = yaml.load(fh)

    def data(self):
        return self._config_data
