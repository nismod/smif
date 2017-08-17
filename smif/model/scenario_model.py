from smif.metadata import MetadataSet
from smif.model.composite import Model


class ScenarioModel(Model):
    """Represents exogenous scenario data

    Arguments
    ---------
    name : string
        The unique name of this scenario
    output : smif.metadata.MetaData
        A name for the scenario output parameter
    """

    def __init__(self, name, output=None):
        if output:
            if isinstance(output, MetadataSet):
                super().__init__(name)
                self._model_outputs = output
            else:
                msg = "output argument should be type smif.metadata.MetadataSet"
                raise TypeError(msg)
        else:
            super().__init__(name)

        self._data = {}
        self.timesteps = []

    def add_output(self, name, spatial_resolution, temporal_resolution, units):
        """Add an output to the scenario model

        Arguments
        ---------
        name: str
        spatial_resolution: :class:`smif.convert.area.RegionRegister`
        temporal_resolution: :class:`smif.convert.interval.TimeIntervalRegister`
        units: str

        """
        output_metadata = {"name": name,
                           "spatial_resolution": spatial_resolution,
                           "temporal_resolution": temporal_resolution,
                           "units": units}

        self._model_outputs.add_metadata(output_metadata)

    def add_data(self, data, timesteps):
        """Add data to the scenario

        Arguments
        ---------
        data : dict
            Key of dict should be timestep, and value a numpy array which
            matches the spatial and temporal resolution associated
            with the output
        """
        self.timesteps = timesteps
        self._data = data

    def simulate(self, timestep, data=None):
        """Returns the scenario data
        """
        time_index = self.timesteps.index[timestep]
        return {self.model_outputs.names[0]: self._data[time_index]}
