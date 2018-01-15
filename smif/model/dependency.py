from logging import getLogger

from smif.convert import SpaceTimeConvertor


class Dependency(object):
    """Link a model input to a data source

    Arguments
    ---------
    source_model : smif.model.Model
        The source model object
    source : smif.metadata.Metadata
        The source parameter (output) object
    sink : smif.metadata.Metadata
        The sink parameter (input) object
    function=None : func
        A conversion function
    """
    def __init__(self, source_model, source, sink, function=None):
        self.logger = getLogger(__name__)

        self.source_model = source_model
        self.source = source
        self.sink = sink

        # Insist on identical metadata - conversions to be explicit
        if source.spatial_resolution.name != \
                sink.spatial_resolution.name:
            raise NotImplementedError(
                "Implicit spatial conversion not implemented (attempted {}>{})".format(
                    source.spatial_resolution.name,
                    sink.spatial_resolution.name))
        if source.temporal_resolution.name != \
                sink.temporal_resolution.name:
            raise NotImplementedError(
                "Implicit spatial conversion not implemented (attempted {}>{})".format(
                    source.temporal_resolution.name,
                    sink.temporal_resolution.name))
        if source.units != sink.units:
            raise NotImplementedError(
                "Implicit units conversion not implemented (attempted {}>{})".format(
                    source.units,
                    sink.units))

        if function:
            self.convert = function
        else:
            self.convert = self._default_convert

    def _default_convert(self, data):
        """Convert dependency data

        Arguments
        ---------
        data : numpy.ndarray
            The data series for conversion
        """
        from_units = self.source.units
        to_units = self.sink.units
        self.logger.debug("Unit conversion: %s -> %s", from_units, to_units)

        if from_units != to_units:
            raise NotImplementedError("Units conversion not implemented %s - %s",
                                      from_units, to_units)

        convertor = SpaceTimeConvertor()
        return convertor.convert(data,
                                 self.source.spatial_resolution.name,
                                 self.sink.spatial_resolution.name,
                                 self.source.temporal_resolution.name,
                                 self.sink.temporal_resolution.name)

    def __repr__(self):
        return "<Dependency({}, {}, {})".format(self.source_model, self.source, self.sink)

    def __eq__(self, other):
        return self.source_model == other.source_model \
            and self.source == other.source \
            and self.sink == other.sink
