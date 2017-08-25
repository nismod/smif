from logging import getLogger

from smif.convert import SpaceTimeConvertor


class Dependency():
    """

    Arguments
    ---------
    source_model : smif.composite.Model
        The source model object
    source : smif.metadata.Metadata
        The source parameter (output) object
    function=None : func
        A conversion function
    """

    def __init__(self, source_model, source, function=None):

        self.logger = getLogger(__name__)

        self.source_model = source_model
        self.source = source
        if function:
            self._function = function
        else:
            self._function = self.convert

    def convert(self, data, model_input):

        from_units = self.source.units
        to_units = model_input.units
        self.logger.debug("Unit conversion: %s -> %s", from_units, to_units)

        if from_units != to_units:
            raise NotImplementedError("Units conversion not implemented %s - %s",
                                      from_units, to_units)

        spatial_resolution = model_input.spatial_resolution.name
        temporal_resolution = model_input.temporal_resolution.name
        return self._convert_data(data,
                                  spatial_resolution,
                                  temporal_resolution)
        return data

    def _convert_data(self, data, to_spatial_resolution,
                      to_temporal_resolution):
        """Convert data from one spatial and temporal resolution to another

        Parameters
        ----------
        data : numpy.ndarray
            The data series for conversion
        to_spatial_resolution : smif.convert.register.ResolutionSet
        to_temporal_resolution : smif.convert.register.ResolutionSet

        Returns
        -------
        converted_data : numpy.ndarray
            The converted data series

        """
        convertor = SpaceTimeConvertor()
        return convertor.convert(data,
                                 self.source.spatial_resolution.name,
                                 to_spatial_resolution,
                                 self.source.temporal_resolution.name,
                                 to_temporal_resolution)

    def get_data(self, timestep, model_input):
        data = self.source_model.simulate(timestep)
        return self._function(data[self.source.name], model_input)

    def __repr__(self):
        return "Dependency('{}', '{}')".format(self.source_model.name,
                                               self.source.name)

    def __eq__(self, other):
        return self.source_model == other.source_model \
            and self.source == other.source
