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
    def asset_names(self):
        """Returns the set of unique assets defined in the planning commands
        """
        return set([plan['asset_type'] for plan in self.planning])

    @property
    def timeperiods(self):
        """Returns the set of unique time periods defined in the planning
        commands
        """
        return set([plan['build_date'] for plan in self.planning])
