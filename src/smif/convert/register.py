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


class LogMixin(object):

    @property
    def logger(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)


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

        if self.axis is not None:
            data_count = data.shape[self.axis]
            if coefficients.shape[0] != data_count:
                msg = "Size of coefficient array does not match source " \
                      "resolution set from data matrix: %s != %s"
                raise ValueError(msg, coefficients.shape[self.axis],
                                 data_count)

        if self.axis == 0:
            converted = np.dot(coefficients.T, data)
        elif self.axis == 1:
            converted = np.dot(data, coefficients)
        else:
            converted = np.dot(data, coefficients)

        self.logger.debug("Converting from %s to %s.", from_set_name, to_set_name)
        self.logger.debug("Converted value from %s to %s", data.sum(), converted.sum() )

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
            self.data_interface.write_coefficients(source,
                                                   destination,
                                                   data)
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

        if self.data_interface:

            self.logger.info("Using data interface to load coefficients")
            coefficients = self.data_interface.read_coefficients(source,
                                                                 destination)

            if coefficients is None:
                msg = "Coefficients not found, generating coefficients for %s to %s"
                self.logger.info(msg, source, destination)

                coefficients = self._obtain_coefficients(from_set, to_set)
                self._write_coefficients(source, destination, coefficients)

        else:

            msg = "No data interface specified, generating coefficients for %s to %s"
            self.logger.info(msg, source, destination)
            coefficients = self._obtain_coefficients(from_set, to_set)

        return coefficients

    def _obtain_coefficients(self, from_set, to_set):
        """
        """
        coefficients = np.zeros((len(from_set), len(to_set)), dtype=np.float)
        self.logger.debug("Coefficients array is of shape %s for %s to %s",
                          coefficients.shape, from_set.name, to_set.name)

        from_names = from_set.get_entry_names()
        for to_idx, to_entry in enumerate(to_set):
            for from_idx in from_set.intersection(to_entry):

                proportion = from_set.get_proportion(from_idx, to_entry)

                self.logger.debug("%i percent of %s is in %s",
                                  proportion * 100,
                                  to_entry.name, from_set.data[from_idx].name)
                from_idx = from_names.index(from_set.data[from_idx].name)

                coefficients[from_idx, to_idx] = proportion
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
