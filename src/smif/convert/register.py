"""Register, ResolutionSet abstract classes to contain metadata and generate conversion
coefficients.

NDimensionalRegister is used in :class:`smif.convert.interval.IntervalAdaptor` and
:class:`smif.convert.region.RegionAdaptor`.
"""
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import numpy as np


class LogMixin(object):

    @property
    def logger(self):
        try:
            logger = self._logger
        except AttributeError:
            name = '.'.join([__name__, self.__class__.__name__])
            logger = logging.getLogger(name)
            self._logger = logger
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger


class Register(LogMixin, metaclass=ABCMeta):
    """Abstract class which holds the ResolutionSets

    Arguments
    ---------
    axis : int, default=None
        The axis over which operations on the data array are performed

    """
    data_interface = None

    def __init__(self, axis=None):
        self.axis = axis

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

    def convert(self, data, from_set_name, to_set_name):
        """Convert a list of data points for a given set to another set

        .. deprecated
                Usage superceded by Adaptor.convert

        Parameters
        ----------
        data: numpy.ndarray
        from_set_name: str
        to_set_name: str

        Returns
        -------
        numpy.ndarray

        """
        coefficients = self.get_coefficients(from_set_name, to_set_name)
        converted = Register.convert_with_coefficients(data, coefficients, self.axis)

        self.logger.debug("Converting from %s to %s.", from_set_name, to_set_name)
        self.logger.debug("Converted value from %s to %s", data.sum(), converted.sum())

        return converted

    @staticmethod
    def convert_with_coefficients(data, coefficients, axis=None):
        """Convert an array of data using given coefficients, along a given axis

        .. deprecated
                Usage superceded by Adaptor.convert

        Parameters
        ----------
        data: numpy.ndarray
        coefficients: numpy.ndarray
        axis: integer, optional

        Returns
        -------
        numpy.ndarray

        """
        if axis is not None:
            data_count = data.shape[axis]
            if coefficients.shape[0] != data_count:
                msg = "Size of coefficient array does not match source " \
                      "resolution set from data matrix: %s != %s"
                raise ValueError(msg, coefficients.shape[axis], data_count)

        if axis == 0:
            converted = np.dot(coefficients.T, data)
        elif axis == 1:
            converted = np.dot(data, coefficients)
        else:
            converted = np.dot(data, coefficients)

        return converted


class NDimensionalRegister(Register):
    """Abstract class which holds N-Dimensional ResolutionSets

    Arguments
    ---------
    axis : int, default=None
        The axis over which operations on the data array are performed

    """
    def __init__(self, axis=None):
        super().__init__(axis)
        self._register = OrderedDict()
        self._conversions = defaultdict(dict)

    def register(self, resolution_set):
        """Add a ResolutionSet to the register

        Parameters
        ----------
        resolution_set : :class:`smif.convert.ResolutionSet`

        Raises
        ------
        ValueError
            If a ResolutionSet of the same name already exists in the register
        """
        if resolution_set.name in self._register:
            msg = "A ResolutionSet named {} has already been loaded"
            raise ValueError(msg.format(resolution_set.name))

        self.logger.info("Registering '%s' with %i items",
                         resolution_set.name,
                         len(resolution_set))

        self._register[resolution_set.name] = resolution_set

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

    def _write_coefficients(self, source, destination, data):
        if self.data_interface:
            self.data_interface.write_coefficients(source, destination, data)
        else:
            msg = "Data interface not available to write coefficients"
            self.logger.warning(msg)

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

        if from_set.coverage != to_set.coverage:
            log_msg = "Coverage for '%s' is %d and does not match coverage " \
                    "for '%s' which is %d"
            self.logger.warning(log_msg, from_set.name, from_set.coverage,
                                to_set.name, to_set.coverage)

        coefficients = self.generate_coefficients(from_set, to_set)

        return coefficients

    def generate_coefficients(self, from_set, to_set):
        """Generate coefficients for converting between two :class:`ResolutionSet`s

        Coefficients for converting a single dimension will always be 2D, of shape
        (len(from_set), len(to_set)).

        Parameters
        ----------
        from_set : ResolutionSet
        to_set : ResolutionSet

        Returns
        -------
        numpy.ndarray
        """
        coefficients = np.zeros((len(from_set), len(to_set)), dtype=np.float)
        self.logger.debug("Coefficients array is of shape %s for %s to %s",
                          coefficients.shape, from_set.name, to_set.name)

        from_names = from_set.get_entry_names()
        for to_idx, to_entry in enumerate(to_set):
            for from_idx in from_set.intersection(to_entry):
                from_entry = from_set.data[from_idx]
                proportion = from_set.get_proportion(from_idx, to_entry)

                self.logger.debug("%i percent of %s (#%s) is in %s (#%s)",
                                  proportion * 100,
                                  to_entry.name, to_idx,
                                  from_entry.name, from_idx)
                from_idx = from_names.index(from_entry.name)

                coefficients[from_idx, to_idx] = proportion
        self.logger.debug("Generated %s", coefficients)
        return coefficients


class ResolutionSet(metaclass=ABCMeta):
    """Abstract class which holds the Resolution definitions
    """
    def __init__(self):

        self.name = ''
        self.description = ''
        self.data = []
        self.logger = logging.getLogger(__name__)

    def as_dict(self):
        """Get a serialisable representation of the object
        """
        return {'name': self.name,
                'description': self.description}

    def __iter__(self):
        return iter(self.data)

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

    @abstractmethod
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

    @property
    @abstractmethod
    def coverage(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_bounds(entry):
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
