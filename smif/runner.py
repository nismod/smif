import logging

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
    def __init__(self, model_name, attributes):
        """
        Arguments
        =========
        project_folder: str
            The path to the project folder
        model_name : str
            Name of the model
        """
        self._model_name = model_name
        self._attributes = attributes

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
        return sorted([asset for asset in self._attributes.keys()])

    @property
    def attributes(self):
        """The collection of asset attributes

        Returns
        =======
        dict
            The collection of asset attributes
        """
        return self._attributes
