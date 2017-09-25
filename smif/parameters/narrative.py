"""
"""


class NarrativeData(object):
    """Holds information relating to parameters

    Arguments
    ---------
    name : str
    description :str
    filename : str
    narrative_set : str

    Example
    --------
    >>> narrative_data = {'name': 'parameter_name',
                            'value': 42}
    >>> narrative = NarrativeData('Energy Demand - High Tech',
                                    'A description',
                                    'energy_demand_high_tech.yml',
                                    'technology')
    """
    def __init__(self, name, description, filename, narrative_set):
        self._name = name
        self._description = description
        self._filename = filename
        self._narrative_set = narrative_set

        self._data = None

    def as_dict(self):
        """Serialise the narrative data

        """
        config = {'name': self._name,
                  'description': self._description,
                  'filename': self._filename,
                  'narrative_set': self._narrative_set}
        return config

    def add_data(self, data):
        """Add data to the NarrativeData object

        Arguments
        ---------
        data : dict
            A dictionary of overridden parameter values

        Example
        -------
        >>> narrative_data = {'name': 'parameter_name',
                              'value': 42}
        >>> narrative = NarrativeData('Energy Demand - High Tech',
                                      'A description',
                                      'energy_demand_high_tech.yml',
                                      'technology')
        >>> narrative.add_data(narrative_data)
        """
