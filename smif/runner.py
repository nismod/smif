import logging
import os
from smif.parse_config import ConfigParser

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class ModelRunner(object):
    """
    """
    def __init__(self, project_folder, model_name):
        """
        Arguments
        =========
        project_folder: str
            The path to the project folder
        model_name : str
            Name of the model
        """
        self._project_folder = project_folder
        self._model_name = model_name

        self._assets = self.load_assets()

    @property
    def name(self):
        return self._model_name

    @property
    def assets(self):
        return self._assets

    def load_assets(self):
        """Loads the assets from the sector model folders

        Using the list of model folders extracted from the configuration file,
        this function returns a list of all the assets from the sector models

        Returns
        =======
        list
            A list of assets from all the sector models


        Notes
        =====
        This should really be pushed into a SectorModel class, with a list of
        assets generated for the sos on demand

        """
        assets = []
        path_to_assetfile = os.path.join(self._project_folder,
                                         'models',
                                         self._model_name,
                                         'assets')

        for assetfile in os.listdir(path_to_assetfile):
            asset_path = os.path.join(path_to_assetfile, assetfile)
            logger.info("Loading assets from {}".format(asset_path))
            assets.extend(ConfigParser(asset_path).data)
        return assets
