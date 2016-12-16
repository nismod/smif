"""The decision module handles the three planning levels

Currently, only pre-specified planning is implemented.

"""


class Planning:
    """

    Arguments
    =========
    planning_data : dict
        A dictionary of pre-specified planning commands
    """

    def __init__(self, planning_data):
        if planning_data is not None:
            self.planning = planning_data
        else:
            self.planning = []

    @property
    def assets(self):
        """Returns the set of unique assets defined in the planning commands
        """
        return set([plan['asset'] for plan in self.planning])

    @property
    def timeperiods(self):
        """Returns the set of unique time periods defined in the planning
        commands
        """
        return set([plan['timeperiod'] for plan in self.planning])
