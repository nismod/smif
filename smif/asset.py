"""Asset is the general term for any component of an infrastructure system

The set of assets defines the state of the infrastructure system.

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
        self.data = data # should behave as key=>value dict

        if "sector" not in data:
            # sector is required, may be None
            self.sector = None

    def sha1sum(self):
        str_to_hash = str(self).encode('utf-8')
        return hashlib.sha1(str_to_hash).hexdigest()

    def __repr__(self):
        data_str = Asset.deterministic_dict_to_str(self.data)
        return "Asset(\"{}\", {})".format(self.asset_type, data_str)

    def __str__(self):
        return Asset.deterministic_dict_to_str(self.data)

    @staticmethod
    def deterministic_dict_to_str(data):
        return json.dumps(data, sort_keys=True)

    @property
    def sector(self):
        return self.data["sector"]

    @sector.setter
    def sector(self, value):
        self.data["sector"] = value

    @property
    def build_date(self):
        if "build_date" not in self.data:
            return None
        return self.data["build_date"]

    @build_date.setter
    def build_date(self, value):
        self.data["build_date"] = value

    @property
    def location(self):
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

    ## Internal data structures

    `asset_types` is a 2D array of integers: each entry is an array representing
    an asset type, each integer indexes attribute_possible_values

    `attribute_keys` is a 1D array of strings

    `attribute_possible_values` is a 2D array of simple values, possibly
    (boolean, integer, float, string, tuple). Each entry is a list of possible
    values for the attribute at that index.

    ## Invariants

    - there must be one name and one list of possible values per attribute
    - each asset type must list one value for each attribute, and that
      value must be a valid index into the possible_values array
    - each possible_values array should be all of a single type
    """
    def __init__(self):
        self.asset_types = []
        self.attribute_keys = []
        self.attribute_possible_values = []

    def register(self, asset):
        for key, value in asset.data.items():
            self.register_attribute(key, value)

        numeric_asset = [0] * len(self.attribute_keys)

        for key, value in asset.data.items():
            attr_idx = self.attribute_index(key)
            value_idx = self.attribute_value_index(attr_idx, value)
            numeric_asset[attr_idx] = value_idx

        self.asset_types.append(numeric_asset)

    def register_attribute(self, key, value):
        if key not in self.attribute_keys:
            self.attribute_keys.append(key)
            self.attribute_possible_values.append([None])

        attr_idx = self.attribute_index(key)

        if value not in self.attribute_possible_values[attr_idx]:
            self.attribute_possible_values[attr_idx].append(value)

    def attribute_index(self, key):
        return self.attribute_keys.index(key)

    def attribute_value_index(self, attr_idx, value):
        return self.attribute_possible_values[attr_idx].index(value)

    def numeric_to_asset(self, numeric_asset):
        asset = Asset()
        for attr_idx, value_idx in enumerate(numeric_asset):
            key = self.attribute_keys[attr_idx]
            value = self.attribute_possible_values[attr_idx][value_idx]

            if key == "asset_type":
                asset.asset_type = value

            asset.data[key] = value

        return asset

