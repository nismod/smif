"""Read and parse the config for sector models
"""
import logging
import os
from glob import glob
from . parse_config import ConfigParser

LOGGER = logging.getLogger(__name__)

class SectorModelReader(object):
    """Parses the configuration and input data for a sector model

    Arguments
    =========
    model_name : str
        The name of the model
    model_path : str
        The path to the python module file that contains an implementation of SectorModel
    model_config_dir : str
        The root path of model config/data to use

    """
    def __init__(self, model_name, model_config_dir, model_classname):
        self.model_name = model_name
        self.model_classname = model_classname
        self.model_config_dir = model_config_dir

        self.inputs = None
        self.outputs = None
        self.assets = None

    def load(self):
        """Load and check all config
        """
        self.inputs = self._load_inputs()
        self.outputs = self._load_outputs()
        self.assets = self._load_model_assets()

    def _load_inputs(self):
        """Input spec is located in the ``models/<sectormodel>/inputs.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'inputs.yaml')
        return ConfigParser(path).data

    def _load_outputs(self):
        """Output spec is located in ``models/<sectormodel>/output.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'outputs.yaml')
        return ConfigParser(path).data

    def _load_model_assets(self):
        """Loads the assets from the sector model config folder

        Returns
        =======
        list
            A list of assets for the sector model

        """
        assets = []
        path = os.path.join(self.model_config_dir, 'assets', '*.yaml')

        for asset_path in glob(path):
            LOGGER.info("Loading assets from %s", asset_path)
            file_assets = ConfigParser(asset_path).data
            assets.extend(file_assets)

        return assets
