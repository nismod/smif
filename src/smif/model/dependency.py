from logging import getLogger

from smif.convert import SpaceTimeUnitConvertor


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
            self.logger.warn(
                "Implicit spatial conversion attempted ({}>{})".format(
                    source.spatial_resolution.name,
                    sink.spatial_resolution.name))
        if source.temporal_resolution.name != \
                sink.temporal_resolution.name:
            self.logger.warn(
                "Implicit spatial conversion attempted ({}>{})".format(
                    source.temporal_resolution.name,
                    sink.temporal_resolution.name))

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
        from_spatial = self.source.spatial_resolution.name
        to_spatial = self.sink.spatial_resolution.name
        from_temporal = self.source.temporal_resolution.name
        to_temporal = self.sink.temporal_resolution.name
        from_units = self.source.units
        to_units = self.sink.units

        if from_spatial != to_spatial \
                or from_temporal != to_temporal \
                or from_units != to_units:

            self.logger.debug("SpaceTimeUnit conversion: %s -> %s, %s -> %s, %s -> %s",
                              from_spatial, to_spatial, from_temporal, to_temporal,
                              from_units, to_units)
            convertor = SpaceTimeUnitConvertor()
            data = convertor.convert(data,
                                     from_spatial, to_spatial,
                                     from_temporal, to_temporal,
                                     from_units, to_units)

        return data

    def __repr__(self):
        return "<Dependency({}, {}, {})".format(self.source_model, self.source, self.sink)

    def __eq__(self, other):
        return self.source_model == other.source_model \
            and self.source == other.source \
            and self.sink == other.sink
