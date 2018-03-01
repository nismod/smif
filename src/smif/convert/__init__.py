"""In this module, we implement the conversion across space and time

The :class:`SpaceTimeConvertor` is instantiated with data to convert,
and the names of the four source and destination spatio-temporal resolutions.

The :meth:`~SpaceTimeConvertor.convert` method returns a new
:class:`numpy.ndarray` for passing to a sector model.
"""
import logging
import numpy as np

from smif.convert.area import get_register as get_region_register
from smif.convert.interval import get_register as get_interval_register
from smif.convert.unit import get_register as get_unit_register

__author__ = "Will Usher, Tom Russell, Roald Schoenmakers"
__copyright__ = "Will Usher, Tom Russell, Roald Schoenmakers"
__license__ = "mit"


class Convertor(object):
    """
    """
    def __init__(self):
        self._convertor = SpaceTimeUnitConvertor()
        self.regions = self._convertor.regions
        self.intervals = self._convertor.intervals
        self.units = self._convertor.units

    def convert(self, data, dependency):
        """

        Arguments
        ---------
        data : numpy.ndarray
            Two dimensional array of data indexed over space and time
        dependency : smif.model.Dependency
            The dependency that the data needs to traverse

        1. Looks up coefficient matrix for each from/to pair across dimensions
        2. Performs numpy dot product
        """
        from_spatial = dependency.source.spatial_resolution.name
        to_spatial = dependency.sink.spatial_resolution.name
        from_temporal = dependency.source.temporal_resolution.name
        to_temporal = dependency.sink.temporal_resolution.name
        from_units = dependency.source.units
        to_units = dependency.sink.units

        # a numpy array of dimensions from-by-to regions
        spatial_coefficients = self.regions.get_coefficients(from_spatial, to_spatial)
        # a numpy array of dimensions from-by-to intervals
        temporal_coefficients = self.intervals.get_coefficients(from_temporal, to_temporal)
        # a scalar
        unit_coefficients = self.units.get_coefficients(from_units, to_units)

        converted_data = self.perform_conversion(
            data,
            spatial_coefficients,
            temporal_coefficients,
            unit_coefficients)

        return converted_data

    def perform_conversion(self, data,
                           spatial_coefficients,
                           temporal_coefficients,
                           unit_coefficients):
        """

        Arguments
        ---------
        data : numpy.ndarray
            Holds the data we wish to transform where row :math:`m` is the
            number of source regions and column :math:`n` is the number of
            source intervals. Each entry :math:`x_{ij}` holds a value which is
            mapped to a single region and interval.
        spatial_coefficients : numpy.ndarray
            Contains the weights which attribute the values along the
            source spatial dimension row :math:`m` to those in the target spatial
            dimension column :math:`p`
        temporal_coefficients : numpy.ndarray
            Contains the weights which attribute the values along the source
            temporal dimension row :math:`n` to those in the target temporal
            dimension column :math:`q`.
        unit_coefficients : float
            Contains the scalar value for unit conversion from source to
            destination units

        Returns
        -------
        numpy.ndarray
            The target data matrix :math:`X'=[x'_{kl}]_{(pq)}` holds the
            transformed data in the target spatiotemporal resolution,
            where row :math:`p` is the number of target regions and column
            :math:`q` is the number of target intervals.

        Examples
        --------
        Perform disaggregation over space dimension only:

        >>> data = np.array([[1]])
        >>> space = np.array([[0.333, 0.333, 0.333]])
        >>> time = np.array([[1]])
        >>> unit_coefficients = 1000
        >>> convertor.perform_conversion(data,
                                         space,
                                         time,
                                         unit_coefficients)
        np.array([[333], [333], [333]])

        Perform disaggregation over time dimension only:

        >>> data = np.array([[1]])
        >>> space = np.array([[1.0]])
        >>> time = np.array([[0.333, 0.333, 0.333]])
        >>> unit_coefficients = 1000
        >>> convertor.perform_conversion(data,
                                         space,
                                         time,
                                         unit_coefficients)
        np.array([[333, 333, 333]])

        Perform disaggregation over space and time dimensions:

        >>> data = np.array([[1]])
        >>> space = np.array([[0.333333, 0.333333, 0.333333]])
        >>> time = np.array([[0.333, 0.333, 0.333]])
        >>> unit_coefficients = 1000
        >>> convertor.perform_conversion(data,
                                         space,
                                         time,
                                         unit_coefficients)
        np.array([[111, 111, 111],
                  [111, 111, 111],
                  [111, 111, 111]]))

        Perform aggregation over space and time dimensions:

        >>> data = np.array([[333.333, 333.333, 333.333],
                             [333.333, 333.333, 333.333]])
        >>> space = np.array([[1], [1]])
        >>> time = np.array([[1], [1], [1]])
        >>> unit_coefficients = 1e-3
        >>> convertor.perform_conversion(data,
                                         space,
                                         time,
                                         unit_coefficients)
        np.array([[2]]))

        """
        source_regions, source_intervals = data.shape
        if spatial_coefficients.shape[0] != source_regions:
            msg = "Region counts of spatial coefficients does not match source \
                   regions from data matrix: %s != %s"
            raise ValueError(msg, spatial_coefficients.shape[0],
                             source_regions)
        if temporal_coefficients.shape[0] != source_intervals:
            msg = "Interval counts of temporal coefficients does not match \
                   source intervals from data matrix: %s != %s"
            raise ValueError(msg, temporal_coefficients.shape[0],
                             source_intervals)

        a = np.dot(spatial_coefficients.T, data)
        return np.dot(a, temporal_coefficients) * unit_coefficients

    def compute_intersection(self, source, destination):
        """Compute intersection of source and destination arrays

        Given two arrays containing the areas or durations for source
        and destination compute the proportion of source contained
        in destination

        Arguments
        ---------
        source_array: numpy.ndarray
        destination_array: numpy.ndarray

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        For each element in the source array, find the weighted proportion of
        the destination element.

        """
        assert source.sum(axis=0) == destination.sum(axis=0)

        coefficients = np.zeros((
                                 len(destination),
                                 len(source)
                                 )
                                )

        for d_cell in destination.flat:
            for s_cell in source.flat:
                if d_cell > 0:
                    coefficients = s_cell / d_cell

                    d_cell = d_cell - s_cell
                else:
                    break

        return coefficients


class SpaceTimeUnitConvertor(object):
    """Handles the conversion of time and space for a list of values

    Notes
    -----
    Future development requires using a data object which allows multiple views
    upon the values across the three dimensions of time, space and units. This
    will then allow more efficient conversion across any one of these dimensions
    while holding the others constant.  One option could be
    :class:`collections.ChainMap`.

    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.regions = get_region_register()
        self.intervals = get_interval_register()
        self.units = get_unit_register()

    def convert(self, data,
                from_spatial, to_spatial,
                from_temporal, to_temporal,
                from_unit, to_unit):
        """Convert the data from set of regions and intervals to another

        Parameters
        ----------
        data: numpy.ndarray
            An array of values with dimensions regions x intervals
        from_spatial: str
            The name of the spatial resolution of the data
        to_spatial: str
            The name of the required spatial resolution
        from_temporal: str
            The name of the temporal resolution of the data
        to_temporal: str
            The name of the required temporal resolution
        from_unit: str
            The name of the unit of the data
        to_unit: str
            The name of the required unit

        Returns
        -------
        numpy.ndarray
            An array of data with dimensions regions x intervals
        """
        assert from_spatial in self.regions.names, \
            "Cannot convert from spatial resolution {}".format(from_spatial)
        assert to_spatial in self.regions.names, \
            "Cannot convert to spatial resolution {}".format(to_spatial)
        assert from_temporal in self.intervals.names, \
            "Cannot convert from temporal resolution {}".format(from_temporal)
        assert to_temporal in self.intervals.names, \
            "Cannot convert to temporal resolution {}".format(to_temporal)
        assert from_unit in self.units.names, \
            "Cannot convert from unit {}".format(from_unit)
        assert to_unit in self.units.names, \
            "Cannot convert to unit {}".format(to_unit)

        if from_spatial != to_spatial and from_temporal != to_temporal:
            converted = self._convert_regions(
                self._convert_intervals(
                    data,
                    from_temporal,
                    to_temporal
                ),
                from_spatial,
                to_spatial
            )
        elif from_temporal != to_temporal:
            converted = self._convert_intervals(
                data,
                from_temporal,
                to_temporal
            )
        elif from_spatial != to_spatial:
            converted = self._convert_regions(
                data,
                from_spatial,
                to_spatial
            )
        else:
            converted = data

        if from_unit != to_unit:
            converted = self._convert_units(converted,
                                            from_unit, to_unit)

        return converted

    def _convert_regions(self, data, from_spatial, to_spatial):
        """Slice, convert and compose regions
        """
        converted = np.apply_along_axis(self.regions.convert, 0, data,
                                        from_spatial, to_spatial)
        return converted

    def _convert_intervals(self, data, from_temporal, to_temporal):
        """Slice, convert and compose intervals
        """
        converted = np.apply_along_axis(self.intervals.convert, 1, data,
                                        from_temporal, to_temporal)
        return converted

    def _convert_units(self, data, from_unit, to_unit):
        return self.units.convert(data, from_unit, to_unit)
