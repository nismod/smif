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
