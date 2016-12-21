# -*- coding: utf-8 -*-
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
    model_classname : str
        The name of the class that implements SectorModel
    model_config_dir : str
        The root path of model config/data to use

    """
    def __init__(self, model_name, model_path, model_classname, model_config_dir):
        self.model_name = model_name
        self.model_path = model_path
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

    @property
    def data(self):
        return {
            "name": self.model_name,
            "path": self.model_path,
            "classname": self.model_classname,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "assets": self.assets
        }

    def _load_inputs(self):
        """Input spec is located in the ``data/<sectormodel>/inputs.yaml`` file
        """
        path = os.path.join(self.model_config_dir, 'inputs.yaml')
        return ConfigParser(path).data

    def _load_outputs(self):
        """Output spec is located in ``data/<sectormodel>/output.yaml`` file
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


class AssetList:
    """List of assets to be loaded from files

    - The set of assets (power stations etc.) should be explicitly declared
    in a yaml file.
    - Assets are associated with sector models, not the integration configuration.
    - Assets should be stored in a sub-folder associated with the sector model
    name.
    """

    # TODO use or integrate this
    def __init__(self, filepath):
        self._asset_list = ConfigParser(filepath)
        self._validate({
            "type": "array",
            "uniqueItems": True
        })
        self._asset_attributes = None

    def _validate(self, schema):
        self._asset_list.validate(schema)

    @property
    def asset_list(self):
        return self._asset_list.data

    @property
    def asset_attributes(self):
        return self._asset_attributes.data

    def load_attributes(self, filepath):
        """
        """
        self._asset_attributes = ConfigParser(filepath)
        # TODO drop this or move to json file in schema directory
        schema = {
            "type": "array",
            "oneof": self.asset_list,
            "properties": {
                "cost": {
                    "properties": {
                        "value": {
                            "type": "number",
                            "minimum": 0,
                            "exclusiveMinimum": True
                        },
                        "unit": {
                            'type': 'string'
                        }
                    }
                }
            }
        }
        self._validate(schema)
