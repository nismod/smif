"""Asset is the general term for any component of an infrastructure system

The set of assets defines the state of the infrastructure system.

Notes
-----

This module needs to support:

- initialisation of set of assets from model config (e.g. set of text files;
  database)
  - hold generic list of key/values
- creation of new assets by decision logic (rule-based/optimisation solver)
  - maintain or derive set of possible assets
    - hence distinction between known-ahead values and build-time values. At
    least location and date are specified at build time, possibly also cost,
    capacity as functions of time and location.
- serialisation for passing to models
  - ease of access to full generic data structure
- output list of assets for reporting
  - write out with legible or traceable keys and units for verification and understanding
"""
import hashlib
import json


class Asset(object):
    """An asset.

    An asset's data is set up to be a flexible, plain data structure.

    Parameters
    ----------
    asset_type : str, default=""
        The type of asset, which should be unique across all sectors
    data : dict, default=None
        The dictionary of asset attributes
    """
    def __init__(self, asset_type="", data=None):
        if data is None:
            data = {}

        if asset_type == "" and "asset_type" in data:
            # allow data to set asset_type if none given
            asset_type = data["asset_type"]
        else:
            # otherwise rely on asset_type arg
            data["asset_type"] = asset_type

        self.asset_type = asset_type
        self.data = data

        if "sector" not in data:
            # sector is required, may be None
            self.sector = None

    def sha1sum(self):
        """Compute the SHA1 hash of this asset's data
        """
        str_to_hash = str(self).encode('utf-8')
        return hashlib.sha1(str_to_hash).hexdigest()

    def __repr__(self):
        data_str = Asset.deterministic_dict_to_str(self.data)
        return "Asset(\"{}\", {})".format(self.asset_type, data_str)

    def __str__(self):
        return Asset.deterministic_dict_to_str(self.data)

    @staticmethod
    def deterministic_dict_to_str(data):
        """Return a reproducible string representation of any dict
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

    @property
    def location(self):
        """The location of this asset instance (if specified - asset types
        may not have explicit locations)
        """
        if "location" not in self.data:
            return None
        return self.data["location"]

    @location.setter
    def location(self, value):
        self.data["location"] = value


class AssetRegister(object):
    """Controls asset serialisation to/from numeric representation

    - register each asset type
    - translate a set of assets representing an initial system into numeric
      representation
    - translate a set of numeric actions (e.g. from optimisation routine) into
      Asset objects with human-readable key-value pairs

    Possible responsibility of another class:
    - output a complete list of asset build possibilities (asset type at location)
    - which may then be reduced subject to constraints

    Internal data structures
    ------------------------

    `asset_types` is a 2D array of integers: each entry is an array representing
    an asset type, each integer indexes attribute_possible_values

    `attribute_keys` is a 1D array of strings

    `attribute_possible_values` is a 2D array of simple values, possibly
    (boolean, integer, float, string, tuple). Each entry is a list of possible
    values for the attribute at that index.

    Invariants
    ----------

    - there must be one name and one list of possible values per attribute
    - each asset type must list one value for each attribute, and that
      value must be a valid index into the possible_values array
    - each possible_values array should be all of a single type
    """
    def __init__(self):
        self._asset_types = []
        self._attribute_keys = []
        self._attribute_possible_values = []

    def register(self, asset):
        """Add a new asset to the register
        """
        for key, value in asset.data.items():
            self.register_attribute(key, value)

        numeric_asset = [0] * len(self._attribute_keys)

        for key, value in asset.data.items():
            attr_idx = self.attribute_index(key)
            value_idx = self.attribute_value_index(attr_idx, value)
            numeric_asset[attr_idx] = value_idx

        self._asset_types.append(numeric_asset)

    def register_attribute(self, key, value):
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
        return self._attribute_keys.index(key)

    def attribute_value_index(self, attr_idx, value):
        """Get the index of a possible value for a given attribute index
        """
        return self._attribute_possible_values[attr_idx].index(value)

    def numeric_to_asset(self, numeric_asset):
        """Convert the numeric representation of an asset back to Asset (with
        legible key/value data)

        Given a (very minimal) possible state of a register:

        >>> register = AssetRegister()
        >>> register._asset_types = [[1,1,1]]
        >>> register._attribute_keys = ["asset_type", "capacity", "sector"]
        >>> register._attribute_possible_values = [
        ...     [None, "water_treatment_plant"],
        ...     [None, {"value": 5, "units": "ML/day"}],
        ...     [None, "water_supply"]
        ... ]

        Calling this function would piece together the asset:

        >>> asset = register.numeric_to_asset([1,1,1])
        >>> print(asset)
        Asset("water_treatment_plant", {"asset_type": "water_treatment_plant",
        "capacity": {"units": "ML/day", "value": 5}, "sector": "water_supply"})

        """
        asset = Asset()
        for attr_idx, value_idx in enumerate(numeric_asset):
            key = self._attribute_keys[attr_idx]
            value = self._attribute_possible_values[attr_idx][value_idx]

            if key == "asset_type":
                asset.asset_type = value

            asset.data[key] = value

        return asset
