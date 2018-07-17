"""This module handles the collection of interventions in a sector.
The set of interventions describes the targets of possible
physical (or non-physical) decisions which can be made in the sector.

Notes
-----

This module implements:

- initialisation of the set of interventions from model config
 (either a collection of yaml text files, or a database)

  - hold generic list of key/values
  - creation of new interventions by decision logic (rule-based/optimisation solver)
  - maintain or derive set of possible interventions
  - makes the distinction between known-ahead values and build-time values.
    Location and date are specified at build time, while cost and capacity
    are a function of time and location.

- serialisation for passing to models

  - ease of access to full generic data structure

- output list of interventions for reporting

  - write out with legible or traceable keys and units for verification and
    understanding

*Terminology*

name:
    An infrastructure intervention (e.g. power station, policy)
    which holds default attribute/value pairs. These names can be
    inherited by intervention/intervention definitions to reduce the degree of
    duplicate data entry.
intervention:
    A potential intervention or investment.
    Interventions are defined in the same way as for interventions,
    cannot have a ``build_date`` defined.

"""
import hashlib
import json

__author__ = "Will Usher, Tom Russell"
__copyright__ = "Will Usher, Tom Russell"
__license__ = "mit"


class Intervention(object):
    """An potential investment to send to the decision manager

    An Intervention, is an investment which has a name (or name),
    other attributes (such as capital cost and economic lifetime),
    and location, but no build date.

    The set of interventions are defined within each sector, and these are
    collected into an :class:`InterventionRegister` when
    a :class:`smif.controller.SosModel` is instantiated by the controller at
    runtime.

    An intervention's data is set up to be a flexible, plain data structure.

    Parameters
    ==========
    name : str, default=""
        The name of the intervention, which should be unique across all sectors
    data : dict, default=None
        The dictionary of intervention attributes
    sector : str, default=""
        The sector associated with the intervention
    location : str, default=None
        The location of the intervention

    """
    def __init__(self, name="", data=None, sector="", location=None):

        if data is None:
            data = {}
        assert isinstance(name, str)

        if name == "" and "name" in data:
            # allow data to set name if none given
            name = data["name"]
        else:
            # otherwise rely on name arg
            data["name"] = name

        if sector == "" and "sector" in data:
            # sector is required, may be None
            sector = data["sector"]
        else:
            data["sector"] = sector

        if location is None and "location" in data:
            # location is required
            location = data["location"]
        else:
            data["location"] = location

        self.data = data
        self.name = name
        self.sector = sector

        assert self._validate(['name', 'location'], ['build_date'])

    def as_dict(self):
        return self.data

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

        return True

    def sha1sum(self):
        """Compute the SHA1 hash of this intervention's data

        Returns
        -------
        str
        """
        str_to_hash = str(self).encode('utf-8')
        return hashlib.sha1(str_to_hash).hexdigest()

    def __repr__(self):
        data_str = self.deterministic_dict_to_str(self.data)
        return "Intervention(\"{}\", {})".format(self.name, data_str)

    def __str__(self):
        return self.deterministic_dict_to_str(self.data)

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
        """The name of the sector model this intervention is used in.
        """
        return self.data["sector"]

    @sector.setter
    def sector(self, value):
        self.data["sector"] = value

    @property
    def location(self):
        """The location of this intervention instance (if specified - intervention types
        may not have explicit locations)
        """
        return self.data["location"]

    @location.setter
    def location(self, value):
        self.data["location"] = value


class InterventionRegister(object):
    """The collection of Intervention objects

    An InterventionRegister contains an immutable collection of sector specific
    interventions and decision points which can be decided on by the Logic Layer

    * Reads in a collection of interventions defined in each sector model

    * Builds an ordered and immutable collection of interventions

    * Provides interfaces to

      * optimisation/rule-based planning logic

      * SectorModel class model wrappers

    Key functions:

    - outputs a complete list of intervention build possibilities (intervention type at
      location) which are (potentially) constrained by the pre-specified
      planning instructions and existing infrastructure.

    - translate a binary vector of build instructions
      (e.g. from optimisation routine) into intervention objects with human-readable
      key-value pairs

    - translates an immutable collection of intervention objects into a binary vector
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
        self._names = {}
        self._numeric_keys = []
        self.interventions = {}
        self._attribute_keys = []
        self._attribute_possible_values = []

    def __iter__(self):
        """Iterate over the list of intervention types held in the register
        """
        for intervention in self._numeric_keys:
            yield self.numeric_to_intervention(intervention)

    def get_intervention(self, name):
        """Returns the named intervention data
        """
        if name in self._names.keys():
            numeric_key = self._names[name]
            return self.numeric_to_intervention(numeric_key)
        else:
            msg = "Intervention '{}' not found in register"
            raise ValueError(msg.format(name))

    def _check_new_intervention(self, intervention):
        """Checks that the intervention doesn't exist in the register

        """
        hash_list = []
        for existing_intervention in self._numeric_keys:
            hash_list.append(self.numeric_to_intervention(existing_intervention).sha1sum())
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

            numeric_intervention = [0] * len(self._attribute_keys)

            for key, value in intervention.data.items():
                attr_idx = self.attribute_index(key)
                value_idx = self.attribute_value_index(attr_idx, value)
                numeric_intervention[attr_idx] = value_idx

            self._numeric_keys.append(numeric_intervention)
            self._names[intervention.name] = numeric_intervention
        else:
            msg = "Attempted registering of duplicate intervention: '{}' for '{}'"
            raise ValueError(msg.format(intervention.name, intervention.sector))

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

    def numeric_to_intervention(self, numeric_intervention):
        """Convert the numeric representation of an intervention back to intervention (with
        legible key/value data)

        Parameters
        ----------
        numeric_intervention : list
            A list of integers of length `self._attribute_keys`

        Returns
        -------
        Intervention
            An :class:`Intervention` object

        Examples
        --------

        Given a (very minimal) possible state of a register:

        >>> register = interventionRegister()
        >>> register._names = [[1,1,1]]
        >>> register._attribute_keys = ["name", "capacity", "sector"]
        >>> register._attribute_possible_values = [
        ...     [None, "water_treatment_plant"],
        ...     [None, {"value": 5, "units": "ML/day"}],
        ...     [None, "water_supply"]
        ... ]

        Calling this function would piece together the intervention:

        >>> intervention = register.numeric_to_intervention([1,1,1])
        >>> print(intervention)
        intervention("water_treatment_plant", {"name": "water_treatment_plant",
        "capacity": {"units": "ML/day", "value": 5}, "sector": "water_supply"})

        """
        data = {}
        for attr_idx, value_idx in enumerate(numeric_intervention):
            key = list(self._attribute_keys)[attr_idx]
            value = self._attribute_possible_values[attr_idx][value_idx]

            data[key] = value

        intervention = Intervention(data=data)

        return intervention

    def __len__(self):
        """Returns the number of intervention types stored in the register
        """
        return len(self._numeric_keys)
