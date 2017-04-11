"""This module handles the collection of interventions and assets in a sector.
The set of interventions describes the targets of possible
physical (or non-physical) decisions which can be made in the sector.

An Asset is the general term for an existing component of an infrastructure system.

The difference between an Intervention and an Asset, is that the latter exists
(it has been "built"), whereas the former describes the potential to build an Asset.

The set of assets defines the 'state' of the infrastructure system.

Notes
-----

This module implements:

- initialisation of the set of assets from model config (either a collection of yaml
  text files, or a database)

  - hold generic list of key/values
  - creation of new assets by decision logic (rule-based/optimisation solver)
  - maintain or derive set of possible assets
  - makes the distinction between known-ahead values and build-time values.
    Location and date are specified at build time, while cost and capacity
    are a function of time and location.

- serialisation for passing to models

  - ease of access to full generic data structure

- output list of assets for reporting

  - write out with legible or traceable keys and units for verification and
    understanding

*Terminology*

name:
    A category of infrastructure intervention (e.g. power station, policy)
    which holds default attribute/value pairs. These names can be
    inherited by asset/intervention definitions to reduce the degree of
    duplicate data entry.
asset:
    An instance of an intervention, which represents a single investment
    decisions which will take place, or has taken place.
    Historical interventions are defined as initial conditions, while
    future interventions are listed as pre-specified planning.
    Both historical and future interventions can make use of names to
    ease data entry.  Assets must have ``location``, ``build_date``
    and ``name`` attributes defined.
intervention:
    A potential asset or investment.
    Interventions are defined in the same way as for assets,
    cannot have a ``build_date`` defined.

"""
import hashlib
import json

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class InterventionContainer(object):
    """An container for asset types, interventions and assets.

    An asset's data is set up to be a flexible, plain data structure.

    Parameters
    ----------
    name : str, default=""
        The type of asset, which should be unique across all sectors
    data : dict, default=None
        The dictionary of asset attributes
    sector : str, default=""
        The sector associated with the asset

    """
    def __init__(self, name="", data=None, sector=""):

        assert isinstance(name, str)

        if data is None:
            data = {}

        if name == "" and "name" in data:
            # allow data to set name if none given
            name = data["name"]
        else:
            # otherwise rely on name arg
            data["name"] = name

        self.name = name
        self.data = data

        if sector == "" and "sector" in data:
            # sector is required, may be None
            sector = data["sector"]
        else:
            data["sector"] = sector

        self.sector = sector

        (required, omitted) = self.get_attributes()
        self._validate(required, omitted)

    def get_attributes(self):
        """Override to return two lists, one containing required attributes,
        the other containing omitted attributes

        Returns
        -------
        tuple
            Tuple of lists, one contained required attributes, the other which
            must be omitted
        """
        raise NotImplementedError

    def _validate(self, required_attributes, omitted_attributes):
        """Ensures location is present and no build date is specified

        """
        keys = self.data.keys()
        for expected in required_attributes:
            if expected not in keys:
                msg = "Validation failed due to missing attribute: '{}' in {}"
                raise ValueError(msg.format(expected, str(self)))

        for omitted in omitted_attributes:
            if omitted in keys:
                msg = "Validation failed due to extra attribute: '{}' in {}"
                raise ValueError(msg.format(omitted, str(self)))

    def sha1sum(self):
        """Compute the SHA1 hash of this asset's data

        Returns
        -------
        str
        """
        str_to_hash = str(self).encode('utf-8')
        return hashlib.sha1(str_to_hash).hexdigest()

    def __repr__(self):
        data_str = Asset.deterministic_dict_to_str(self.data)
        return "Asset(\"{}\", {})".format(self.name, data_str)

    def __str__(self):
        return Asset.deterministic_dict_to_str(self.data)

    @staticmethod
    def deterministic_dict_to_str(data):
        """Return a reproducible string representation of any dict

        Parameters
        ----------
        data : dict
            An intervention attributes dictionary

        Returns
        -------
        str
            A reproducible string representation
        """
        return json.dumps(data, sort_keys=True)

    @property
    def sector(self):
        """The name of the sector model this asset is used in.
        """
        return self.data["sector"]

    @sector.setter
    def sector(self, value):
        self.data["sector"] = value


class Intervention(InterventionContainer):
    """An potential investment to send to the logic-layer

    An Intervention, is an investment which has a name (or name),
    other attributes (such as capital cost and economic lifetime),
    and location, but no build date.

    The set of interventions are defined within each sector, and these are
    collected into an :class:`InterventionRegister` when
    a :class:`smif.controller.SosModel` is instantiated by the controller at
    runtime.

    Parameters
    ==========
    name : str, default=""
        The type of asset, which should be unique across all sectors
    data : dict, default=None
        The dictionary of asset attributes
    sector : str, default=""
        The sector associated with the asset

    """
    def get_attributes(self):
        """Ensures location is present and no build date is specified

        """
        return (['name', 'location'], ['build_date'])

    @property
    def location(self):
        """The location of this asset instance (if specified - asset types
        may not have explicit locations)
        """
        return self.data["location"]

    @location.setter
    def location(self, value):
        self.data["location"] = value


class Asset(Intervention):
    """An instance of an intervention with a build date.

    Used to represent pre-specified planning and existing infrastructure assets
    and interventions

    Parameters
    ----------
    name : str, default=""
        The type of asset, which should be unique across all sectors
    data : dict, default=None
        The dictionary of asset attributes
    sector : str, default=""
        The sector associated with the asset

    """
    def get_attributes(self):
        """Ensures location is present and no build date is specified

        """
        return (['name', 'location', 'build_date'], [])

    @property
    def build_date(self):
        """The build date of this asset instance (if specified - asset types
        will not have build dates)
        """
        if "build_date" not in self.data:
            return None
        return self.data["build_date"]

    @build_date.setter
    def build_date(self, value):
        self.data["build_date"] = value


class Register(object):
    """Holds interventions, pre-spec'd planning instructions & existing assets

    - register each asset type/intervention name
    - translate a set of assets representing an initial system into numeric
      representation

    """
    def __init__(self):
        self.assets = {}
        self._names = []
        self._attribute_keys = []
        self._attribute_possible_values = []

    def register(self, asset):
        """Adds a new asset to the collection
        """
        name = asset.data['name']
        self.assets[name] = asset
        self._names.append(asset.data['name'])

        for key in asset.data.keys():
            self._attribute_keys.append(key)

    def __iter__(self):
        for asset in self.assets:
            yield asset


class AssetRegister(Register):
    """Register each asset type

    """

    def register(self, asset):
        if isinstance(asset, Asset):
            pass
        else:
            msg = "You can only register Assets with this register"
            raise TypeError(msg)
        super().register(asset)

    def __len__(self):
        return len(self._names)


class InterventionRegister(Register):
    """The collection of Intervention objects

    An InterventionRegister contains an immutable collection of sector specific
    assets and decision points which can be decided on by the Logic Layer

    * Reads in a collection of interventions defined in each sector model

    * Builds an ordered and immutable collection of interventions

    * Provides interfaces to

      * optimisation/rule-based planning logic

      * SectorModel class model wrappers

    Key functions:

    - outputs a complete list of asset build possibilities (asset type at
      location) which are (potentially) constrained by the pre-specified
      planning instructions and existing infrastructure.

    - translate a binary vector of build instructions
      (e.g. from optimisation routine) into Asset objects with human-readable
      key-value pairs

    - translates an immutable collection of Asset objects into a binary vector
      to pass to the logic-layer.

    Notes
    =====

    *Internal data structures*

    `Intervention_types`
        is a 2D array of integers: each entry is an array
        representing an Intervention type, each integer indexes
        attribute_possible_values

    `attribute_keys`
        is a 1D array of strings

    `attribute_possible_values`
        is a 2D array of simple values, possibly
        (boolean, integer, float, string, tuple). Each entry is a list of
        possible values for the attribute at that index.

    *Invariants*

    - there must be one name and one list of possible values per attribute

    - each Intervention type must list one value for each attribute, and that
      value must be a valid index into the possible_values array

    - each possible_values array should be all of a single type

    """
    def __init__(self):
        super().__init__()
        self._names = {}
        self._numeric_keys = []

    def get_intervention(self, name):
        """Returns the named asset data
        """
        if name in self._names.keys():
            numeric_key = self._names[name]
            return self.numeric_to_intervention(numeric_key)
        else:
            msg = "Intervention '{}' not found in register"
            raise ValueError(msg.format(name))

    def _check_new_intervention(self, intervention):
        """Checks that the asset doesn't exist in the register

        """
        hash_list = []
        for existing_asset in self._numeric_keys:
            hash_list.append(self.numeric_to_intervention(existing_asset).sha1sum())
        if intervention.sha1sum() in hash_list:
            return False
        else:
            return True

    def register(self, intervention):
        """Add a new intervention to the register

        Parameters
        ----------
        intervention : :class:`Intervention`

        """
        if isinstance(intervention, Intervention):
            pass
        else:
            msg = "You can only register Interventions with this register"
            raise TypeError(msg)

        if self._check_new_intervention(intervention):

            for key, value in intervention.data.items():
                self._register_attribute(key, value)

            numeric_asset = [0] * len(self._attribute_keys)

            for key, value in intervention.data.items():
                attr_idx = self.attribute_index(key)
                value_idx = self.attribute_value_index(attr_idx, value)
                numeric_asset[attr_idx] = value_idx

            self._numeric_keys.append(numeric_asset)
            self._names[intervention.name] = numeric_asset

    def _register_attribute(self, key, value):
        """Add a new attribute and its possible value to the register (or, if
        the attribute has been seen before, add a new possible value)
        """
        if key not in self._attribute_keys:
            self._attribute_keys.append(key)
            self._attribute_possible_values.append([None])

        attr_idx = self.attribute_index(key)

        if value not in self._attribute_possible_values[attr_idx]:
            self._attribute_possible_values[attr_idx].append(value)

    def attribute_index(self, key):
        """Get the index of an attribute name
        """
        return list(self._attribute_keys).index(key)

    def attribute_value_index(self, attr_idx, value):
        """Get the index of a possible value for a given attribute index
        """
        return self._attribute_possible_values[attr_idx].index(value)

    def numeric_to_intervention(self, numeric_asset):
        """Convert the numeric representation of an asset back to Asset (with
        legible key/value data)

        Parameters
        ----------
        numeric_asset : list
            A list of integers of length `self._attribute_keys`

        Returns
        -------
        Intervention
            An :class:`Intervention` object

        Examples
        --------

        Given a (very minimal) possible state of a register:

        >>> register = AssetRegister()
        >>> register._names = [[1,1,1]]
        >>> register._attribute_keys = ["name", "capacity", "sector"]
        >>> register._attribute_possible_values = [
        ...     [None, "water_treatment_plant"],
        ...     [None, {"value": 5, "units": "ML/day"}],
        ...     [None, "water_supply"]
        ... ]

        Calling this function would piece together the asset:

        >>> asset = register.numeric_to_asset([1,1,1])
        >>> print(asset)
        Asset("water_treatment_plant", {"name": "water_treatment_plant",
        "capacity": {"units": "ML/day", "value": 5}, "sector": "water_supply"})

        """
        data = {}
        for attr_idx, value_idx in enumerate(numeric_asset):
            key = list(self._attribute_keys)[attr_idx]
            value = self._attribute_possible_values[attr_idx][value_idx]

            data[key] = value

        intervention = Intervention(data=data)

        return intervention

    def __iter__(self):
        """Iterate over the list of asset types held in the register
        """
        for asset in self._numeric_keys:
            yield self.numeric_to_intervention(asset)

    def __len__(self):
        """Returns the number of asset types stored in the register
        """
        return len(self._numeric_keys)
