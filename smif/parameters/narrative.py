"""Contains classes and methods relating to narratives.

Narratives hold collections of overridden parameter data. During model setup,
a user compiles a narrative file which contains a list of parameter names and
values

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

        self._data = {}

    @property
    def data(self):
        """Returns the narrative data

        Returns
        -------
        dict
            A nested dictionary containing the narrative data::

                {'global': [{'global_parameter': 'value'}],
                 'model_name': [{'model_parameter': 'value'},
                                {'model_parameter_two': 'value'}
                                ]
                }
        """
        return self._data

    def as_dict(self):
        """Serialise the narrative data

        Returns
        -------
        dict
            A dictionary of serialised narrative metadata::

                {'name': 'a_name',
                 'description': 'a description',
                 'filename': 'a filename',
                 'narrative_set': 'a_narrative_set'}

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
        >>> narrative_data = {'global': [{'name': 'parameter_name',
                                          'value': 42}]}
        >>> narrative = NarrativeData('Energy Demand - High Tech',
                                      'A description',
                                      'energy_demand_high_tech.yml',
                                      'technology')
        >>> narrative.add_data(narrative_data)
        """
        if isinstance(data, dict):
            self._data.update(data)
        else:
            raise TypeError("Expected a dict of parameter values")
