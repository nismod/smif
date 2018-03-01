"""Register, NDimensionalRegister and ResolutionSet
abstract classes to contain area, interval and unit metadata.

Implemented by :class:`smif.convert.interval.TimeIntervalRegister`,
:class:`smif.convert.area.RegionRegister` and
:class:`smif.convert.unit.UnitRegister`.
"""
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import numpy as np


class Register(metaclass=ABCMeta):
    """Abstract class which holds the ResolutionSets
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @property
    @abstractmethod
    def names(self):
        raise NotImplementedError

    @abstractmethod
    def register(self, resolution_set):
        raise NotImplementedError

    @abstractmethod
    def get_coefficients(self, source, destination):
        raise NotImplementedError


class NDimensionalRegister(Register):
    """Abstract class which holds N-Dimensional ResolutionSets
    """
    def __init__(self):
        super().__init__()
        self._register = OrderedDict()
        self._conversions = defaultdict(dict)

    @property
    def names(self):
        """Names of registered region sets

        Returns
        -------
        sets: list of str
        """
        return list(self._register.keys())

    def get_entry(self, name):
        """Returns the ResolutionSet of `name`

        Arguments
        ---------
        name : str
            The unique identifier of a ResolutionSet in the register

        Returns
        -------
        smif.convert.ResolutionSet

        """
        if name not in self._register:
            msg = "ResolutionSet '{}' not registered"
            raise ValueError(msg.format(name))
        return self._register[name]

    @abstractmethod
    def get_bounds(self, entry):
        """Implement this helper method to return bounds from an entry in the register

        Arguments
        ---------
        entry
            An entry from a ResolutionSet

        Returns
        -------
        bounds
            The bounds of the entry
        """
        raise NotImplementedError

    def get_proportion(self, entry_a, entry_b):
        """Calculate the proportion of `entry_a` and `entry_b`

        Arguments
        ---------
        entry_a : string
            Name of an entry in `ResolutionSet`
        entry_b : string
            Name of an entry in `ResolutionSet`

        Returns
        -------
        float
            The proportion of `entry_a` and `entry_b`
        """
        raise NotImplementedError

    def get_coefficients(self, source, destination):
        """Get coefficients representing intersection of sets

        Arguments
        ---------
        source : string
            The name of the source set
        destination : string
            The name of the destination set

        Returns
        -------
        numpy.ndarray
        """

        from_set = self.get_entry(source)
        to_set = self.get_entry(destination)

        from_names = from_set.get_entry_names()

        coefficients = np.zeros((len(from_set), len(to_set)), dtype=np.float)

        for to_idx, to_entry in enumerate(to_set):
            bounds = self.get_bounds(to_entry)
            intersecting_from_entries = from_set.intersection(bounds)
            for from_entry in intersecting_from_entries:
                proportion = self.get_proportion(from_entry, to_entry)

                from_idx = from_names.index(from_entry.name)

                coefficients[from_idx, to_idx] = proportion

        return coefficients


class ResolutionSet(metaclass=ABCMeta):
    """Abstract class which holds the Resolution definitions
    """
    def __init__(self):

        self.name = ''
        self.description = ''
        self.data = []

    def as_dict(self):
        """Get a serialisable representation of the object
        """
        return {'name': self.name,
                'description': self.description}

    @abstractmethod
    def get_entry_names(self):
        """Get the names of the entries in the ResolutionSet

        Returns
        -------
        set
            The set of names which identify each entry in the ResolutionSet
        """
        raise NotImplementedError

    @abstractmethod
    def intersection(self, bounds):
        """Return the subset of entries intersecting with the bounds
        """
        raise NotImplementedError
