import logging
import os
from glob import glob

from smif.parse_config import ConfigParser

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class Asset(object):
    """An asset represents a physical structure that persists across timesteps.

    Examples of assets include power stations, water treatment plants, roads,
    railway tracks, airports, ports, centres of demand such as houses or
    factories, waste processing plants etc.

    An Asset is targetted by and influenced by the decision-layer.

    A snapshot of the current set of assets in a model is represented by
    :class:`State` and is persisted across model-years.

    The Asset-state is also persisted (written to the datastore).

    Parameters
    ==========
    name : str
        The name of the asset

    """
    def __init__(self, name, attributes):
        self._name = name
        self._attributes = attributes

    @property
    def name(self):
        """Returns the name of the asset

        Returns
        =======
        str
            The name of the asset
        """
        return self._name

    @property
    def attributes(self):
        """Returns the attributes of the asset

        Returns
        =======
        dict
            The attributes of the asset
        """
        return self._attributes


class ModelRunner(object):
    """Contains the data and methods associated with the smif facing aspects of
    running a sector model

    ModelRunner expects to find a yaml configuration file containing
    - lists of assets in ``models/<model_name>/asset_*.yaml``
    - structure of attributes in ``models/<model_name/<asset_name>.yaml

    ModelRunner expects to find a ``run.py`` file in ``models/<model_name>``.
    ``run.py`` contains a python script which subclasses
    :class:`smif.sectormodel.SectorModel` to wrap the sector model.

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
        """The name of the sector model

        Returns
        =======
        str
            The name of the sector model

        Note
        ====
        The name corresponds to the name of the folder in which the
        configuration is expected to be found

        """
        return self._model_name

    @property
    def assets(self):
        """The names of the assets

        Returns
        =======
        list
            A list of the names of the assets
        """
        asset_names = []
        for asset in self._assets:
            asset_names.append(asset.name)
        return asset_names

    @property
    def attributes(self):
        """The collection of asset attributes

        Returns
        =======
        dict
            The collection of asset attributes
        """
        attributes = {}
        for asset in self._assets:
            attributes[asset.name] = asset.attributes
        return attributes

    def load_assets(self):
        """Loads the assets from the sector model folders

        Using the list of model folders extracted from the configuration file,
        this function returns a list of all the assets from the sector models

        Returns
        =======
        list
            A list of assets from all the sector models

        """
        assets = []
        path_to_assetfile = os.path.join(self._project_folder,
                                         'models',
                                         self._model_name,
                                         'assets',
                                         'asset*')

        for assetfile in glob(path_to_assetfile):
            asset_path = os.path.join(path_to_assetfile, assetfile)
            logger.info("Loading assets from {}".format(asset_path))

            for asset in ConfigParser(asset_path).data:
                attributes = self._load_attributes(asset)
                assets.append(Asset(asset, attributes))
        return assets

    def _load_attributes(self, asset_name):

        model_name = self.name
        project_folder = self._project_folder
        file_path = os.path.join(project_folder, 'models',
                                 model_name, 'assets',
                                 "{}.yaml".format(asset_name))
        attributes = ConfigParser(file_path).data
        return attributes
