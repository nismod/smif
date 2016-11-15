import logging
import os
from smif.parse_config import ConfigParser

__author__ = "Will Usher"
__copyright__ = "Will Usher"
__license__ = "mit"

logger = logging.getLogger(__name__)


class Asset(object):
    """An asset is a decision targeted capacity that persists across timesteps.

    An Asset is otherwise an Input to a model.

    Examples of assets include power stations, water treatment plants, roads,
    railway tracks, airports, ports, centres of demand such as houses or
    factories, waste processing plants etc.

    An Asset is targetted by and influenced by a :class:`Decision` but only
    need to be defined in the :class:`Interface` if targetted
    by a :class:`Decision`.

    A snapshot of the current Assets in a model is represented by
    :class:`State` and is persisted across model-years.

    The Asset-state is also persisted (written to the datastore).

    Parameters
    ==========
    name : str
        The name of the asset

    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """Returns the name of the asset

        Returns
        =======
        str
            The name of the asset
        """
        return self._name


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

            for asset in ConfigParser(asset_path).data:
                assets.append(Asset(asset))
        return assets
