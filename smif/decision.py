"""The decision module handles the three planning levels

Currently, only pre-specified planning is implemented.

"""


class Planning:
    """

    Arguments
    =========
    planning_data : list
        A list of pre-specified planning commands
    """

    def __init__(self, planning_data):
        if planning_data is not None:
            self.build_instructions = planning_data
        else:
            self.build_instructions = []

    @property
    def asset_types(self):
        """Returns the set of assets defined in the planning commands
        """
        return {plan['type'] for plan in self.build_instructions}

    @property
    def timeperiods(self):
        """Returns the set of time periods defined in the planning commands
        """
        return {plan['build_date'] for plan in self.build_instructions}
