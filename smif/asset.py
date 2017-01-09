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

class Asset(object):
    """An asset.

    An asset's data is set up to be a flexible, plain data structure.
    """
    def __init__(self, asset_type, data):
        self.asset_type = asset_type
        self.data = data # should behave as key=>value dict

        if "build_date" not in data:
            self.build_date = None

        if "location" not in data:
            self.location = None

    @property
    def build_date(self):
        return self.data["build_date"]

    @build_date.setter
    def build_date(self, value):
        self.data["build_date"] = value

    @property
    def location(self):
        return self.data["location"]

    @location.setter
    def location(self, value):
        self.data["location"] = value


