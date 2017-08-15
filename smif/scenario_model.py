from smif.composite import Model
from smif.metadata import MetadataSet


class ScenarioModel(Model):
    """Represents exogenous scenario data

    Arguments
    ---------
    name : string
        The unique name of this scenario
    output : smif.metadata.MetaDataSet
        A name for the scenario output parameter
    """

    def __init__(self, name, output):
        assert len(output) == 1
        super().__init__(name, MetadataSet([]), output)
        self._data = []

    def add_data(self, data):
        """Add data to the scenario

        Arguments
        ---------
        data : dict
            Key of dict should be name which matches output name
        """
        self._data = data

    def simulate(self, data=None):
        """Returns the scenario data
        """
        return self._data
