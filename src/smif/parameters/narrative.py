"""Contains classes and methods relating to narratives.

Narrative hold collections of overridden parameter data. During model setup,
a user compiles a number of narrative files which contains a list of parameter
names and values. These are assigned to a narrative set during a model run
and the Narrative object holds this information at runtime.
"""


class Narrative(object):
    """Holds information relating to parameters from a collection of narrative policies

    Arguments
    ---------
    name : str
    description :str
    narrative_set : str

    Example
    --------
    >>> narrative = Narrative('Energy Demand - High Tech',
                                  'A description',
                                  'technology')
    """
    def __init__(self, name, description, narrative_set):
        self._name = name
        self._description = description
        self._narrative_set = narrative_set

        self._data = {}

    @property
    def data(self):
        """The narrative data keyed by model name or ``global``

        Returns
        -------
        dict
            A nested dictionary containing the narrative data::

                {
                    'global': [
                        {'global_parameter': 'value'}
                    ],
                    'model_name': [
                        {'model_parameter': 'value'},
                        {'model_parameter_two': 'value'}
                    ]
                }
        """
        return self._data

    @data.setter
    def data(self, data):
        """Add data to the Narrative object

        Arguments
        ---------
        data : dict
            A dictionary of overridden parameter values

        Example
        -------
        >>> narrative_data = {
                'global': [
                    {'parameter_name': 42}
                ]
            }
        >>> narrative = Narrative(
                'Energy Demand - High Tech',
                'A description',
                'technology')
        >>> narrative.add_data(narrative_data)
        """
        if isinstance(data, dict):
            self._data.update(data)
        else:
            raise TypeError("Expected a dict of parameter values")

    def as_dict(self):
        """Serialise the narrative data

        Returns
        -------
        dict
            A dictionary of serialised narrative metadata::

                {'name': 'a_name',
                 'description': 'a description',
                 'narrative_set': 'a_narrative_set'}

        """
        config = {'name': self._name,
                  'description': self._description,
                  'narrative_set': self._narrative_set}
        return config

    def __eq__(self, other):
        return self.as_dict() == other.as_dict() and \
            self.data == other.data
